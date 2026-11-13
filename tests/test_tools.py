"""The tool layer: notebook, budget, resume, verification, report."""
import pytest

from glassbox.lab import Lab
from glassbox.notebook import Notebook
from glassbox.report import UnbackedClaim
from glassbox.session import Session

from . import synthetic as syn


@pytest.fixture()
def session(tmp_path):
    """A session with the synthetic copy model loaded, as if load_model had been called."""
    s = Session(tmp_path)
    s._open("t")
    s.backend = syn.backend(syn.copy_model())
    s.lab = Lab(s.backend, s._dir())
    s._save_state(backend="local-full", model="synthetic", budget=6)
    return s


def test_notebook_is_append_only_and_tracks_hypothesis_status(tmp_path):
    nb = Notebook(tmp_path / "notebook.jsonl")
    h = nb.append("hypothesis", "head 0.1 does the copying")
    nb.append("note", "ablating 0.1 changes nothing", status="rejected", target=h["id"])
    h2 = nb.append("hypothesis", "head 0.0 does the copying")
    view = {e["id"]: e for e in nb.view()}
    assert view[h["id"]]["status"] == "rejected" and view[h2["id"]]["status"] == "open"
    assert len(nb.entries()) == 3
    assert nb.summary()["hypotheses"]["rejected"][0].startswith(h["id"])


def test_experiments_are_logged_and_budget_is_enforced(session):
    data = syn.copy_data()
    for _ in range(6):
        r = session.run_experiment("baseline", data, exploratory=True)
    assert r["budget"]["remaining"] == 0
    with pytest.raises(RuntimeError, match="budget"):
        session.run_experiment("baseline", data, exploratory=True)
    assert sum(e["kind"] == "experiment" for e in session.notebook.entries()) == 6


def test_results_must_cite_real_experiments(session):
    with pytest.raises(KeyError):
        session.notebook_append("result", "it works", refs=["E0042"])
    with pytest.raises(ValueError):
        session.notebook_append("result", "it works")
    r = session.run_experiment("baseline", syn.copy_data(), exploratory=True)
    assert session.notebook_append("result", "model copies token 0", refs=[r["id"]])["refs"] == [r["id"]]


def test_resume_reads_the_same_notebook(session, tmp_path):
    session.run_experiment("baseline", syn.copy_data(), exploratory=True)
    session.notebook_append("hypothesis", "copying happens in layer 0")
    again = Session(tmp_path).notebook_read("t")
    assert again["summary"]["experiments"] == 1 and len(again["summary"]["hypotheses"]["open"]) == 1
    assert again["model"] == "synthetic" and again["model_in_memory"] is False


COPY_CODE = "def predict(tokens, prompt):\n    return tokens[0]\n"


def test_verification_passes_for_the_true_circuit(session):
    v = session.verify_hypothesis({"name": "copy", **syn.copy_data(40, seed=9), "code": COPY_CODE,
                                   "circuit": {"components": ["0.0"], "mode": "zero"}})
    assert v["status"] == "verified"
    assert v["checks"]["agreement"]["agreement_with_model"] == 1.0
    assert v["checks"]["causal_components"]["circuit_ablated"]["fraction_of_clean"] == pytest.approx(0.0, abs=1e-5)
    assert v["checks"]["causal_components"]["everything_else_ablated"]["fraction_of_clean"] == pytest.approx(1.0)


def test_verification_fails_for_a_wrong_circuit_and_wrong_algorithm(session):
    v = session.verify_hypothesis({"name": "wrong", **syn.copy_data(40, seed=9),
                                   "code": "def predict(tokens, prompt):\n    return tokens[1]\n",
                                   "circuit": {"components": ["0.1"], "mode": "zero"}})
    assert v["status"] == "not-verified"
    assert v["checks"]["agreement"]["pass"] is False and v["checks"]["causal_components"]["pass"] is False


def test_quantized_backends_can_only_screen(session, monkeypatch):
    monkeypatch.setattr(type(session.backend), "name", "local-4bit")
    v = session.verify_hypothesis({"name": "copy", **syn.copy_data(40, seed=9), "code": COPY_CODE,
                                   "circuit": {"components": ["0.0"], "mode": "zero"}})
    assert v["status"] == "screening-pass" and "screening" in v["note"]


def _prereg_baseline(session, threshold=0.9):
    p = session.preregister("the model copies token 0", {"primitive": "baseline", "params": syn.copy_data()},
                            {"metric": "summary.clean_accuracy", "op": ">=", "threshold": threshold, "statement": "accuracy at least 0.9"},
                            ["the model outputs a constant token"])
    return session.run_experiment(prereg_id=p["prereg_id"])


def test_report_refuses_unbacked_claims_and_writes_both_files(session):
    r = _prereg_baseline(session)
    with pytest.raises(UnbackedClaim):
        session.write_report({"claims": [{"text": "trust me", "evidence": []}]})
    with pytest.raises(UnbackedClaim):
        session.write_report({"claims": [{"text": "trust me", "evidence": ["E0999"]}]})
    out = session.write_report({"title": "Copy", "behavior": "copy token 0", "pseudocode": COPY_CODE,
                                "claims": [{"text": "the model copies token 0", "evidence": [r["id"]]}],
                                "circuit": {"nodes": [{"id": "in", "label": "token 0", "layer": 0, "kind": "input"},
                                                      {"id": "h", "label": "head 0.0", "layer": 1}],
                                            "edges": [{"src": "in", "dst": "h", "label": "attend"}]},
                                "key_plots": [r["id"]]})
    md = open(session._dir() / out["markdown"], encoding="utf-8").read()
    assert "full precision (local)" in md and r["id"] in md and "Notebook trail" in md and "Prediction record" in md
    assert "<svg" in open(session._dir() / out["html"], encoding="utf-8").read()
