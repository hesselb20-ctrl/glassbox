"""Pre-registration, prediction scoring and the notebook hash chain."""
import json

import pytest

from glassbox.lab import Lab
from glassbox.notebook import Notebook, resolve, score_prediction
from glassbox.report import UnbackedClaim
from glassbox.session import Session

from . import synthetic as syn


@pytest.fixture()
def session(tmp_path):
    s = Session(tmp_path)
    s._open("t")
    s.backend = syn.backend(syn.copy_model())
    s.lab = Lab(s.backend, s._dir())
    s._save_state(backend="local-full", model="synthetic", budget=20)
    return s


def pre(session, primitive, params, metric, op, threshold, **kw):
    return session.preregister("head 0.0 does the copying", {"primitive": primitive, "params": params},
                               {"metric": metric, "op": op, "threshold": threshold, **kw},
                               ["head 0.1 does the copying"])["prereg_id"]


# ---- scoring ---------------------------------------------------------------------------------------------

REC = {"summary": {"x": 0.42, "top": [{"component": "9.9", "effect": 0.8}, {"component": "9.6", "effect": 0.4}], "name": "a"},
       "data": {"effect": [[0.1, 0.9], [0.0, 0.2]]}}


def test_resolve_paths():
    assert resolve(REC, "summary.x") == 0.42
    assert resolve(REC, "data.effect[0][1]") == 0.9
    assert resolve(REC, "summary.top[component=9.6].effect") == 0.4
    with pytest.raises(KeyError):
        resolve(REC, "summary.missing")


@pytest.mark.parametrize("op,th,verdict", [(">=", 0.4, "confirmed"), (">=", 0.5, "refuted"), ("<", 0.5, "confirmed"),
                                           ("between", [0.4, 0.5], "confirmed"), ("outside", [0.4, 0.5], "refuted")])
def test_score_verdicts(op, th, verdict):
    assert score_prediction({"metric": "summary.x", "op": op, "threshold": th}, REC)["verdict"] == verdict


def test_score_inconclusive_cases():
    assert score_prediction({"metric": "summary.nope", "op": ">=", "threshold": 1}, REC)["verdict"] == "inconclusive"
    assert score_prediction({"metric": "summary.name", "op": ">=", "threshold": 1}, REC)["verdict"] == "inconclusive"
    assert score_prediction({"metric": "summary.x", "op": ">=", "threshold": 0.4, "margin": 0.05}, REC)["verdict"] == "inconclusive"


# ---- the default is: no pre-registration, no run -----------------------------------------------------------

def test_run_without_prereg_is_refused(session):
    with pytest.raises(ValueError, match="pre-registration"):
        session.run_experiment("baseline", syn.copy_data())
    assert session.run_experiment("baseline", syn.copy_data(), exploratory=True)["exploratory"] is True
    assert session.run_experiment("dataset", {"task": "modadd", "n": 4, "p": 7})["exploratory"] is True   # exempt


def test_prereg_runs_exactly_what_was_registered_and_scores_it(session):
    data = syn.copy_data()
    pid = pre(session, "ablation", {**data, "components": ["0.0"], "mode": "zero", "joint": True},
              "summary.fraction_of_clean", "<=", 0.5, statement="zero-ablating 0.0 removes at least half the logit diff")
    with pytest.raises(ValueError, match="pre-registered"):
        session.run_experiment("baseline", data, prereg_id=pid)             # cannot swap the experiment
    r = session.run_experiment(prereg_id=pid)
    assert r["primitive"] == "ablation" and r["prediction_score"]["verdict"] == "confirmed"
    assert r["prediction_score"]["observed"] == pytest.approx(0.0, abs=1e-5)
    with pytest.raises(RuntimeError, match="already been run"):
        session.run_experiment(prereg_id=pid)                               # a prediction is tested once
    on_disk = session.lab.load_experiment(r["id"])
    assert on_disk["prereg_id"] == pid and on_disk["score"]["verdict"] == "confirmed" and on_disk["exploratory"] is False


def test_refuted_predictions_are_counted_and_reported(session):
    data = syn.copy_data()
    wrong = pre(session, "ablation", {**data, "components": ["0.1"], "mode": "zero", "joint": True},
                "summary.fraction_of_clean", "<=", 0.5)
    right = pre(session, "ablation", {**data, "components": ["0.0"], "mode": "zero", "joint": True},
                "summary.fraction_of_clean", "<=", 0.5)
    r1, r2 = session.run_experiment(prereg_id=wrong), session.run_experiment(prereg_id=right)
    assert r1["prediction_score"]["verdict"] == "refuted" and r2["prediction_score"]["verdict"] == "confirmed"
    rec = session.notebook.prediction_record()
    assert (rec["preregistered"], rec["confirmed"], rec["refuted"], rec["hit_rate"]) == (2, 1, 1, 0.5)
    out = session.write_report({"title": "t", "claims": [{"text": "0.0 copies", "evidence": [r2["id"]]},
                                                         {"text": "0.1 does not", "evidence": [r1["id"]]}]})
    md = (session._dir() / out["markdown"]).read_text(encoding="utf-8")
    assert "1 confirmed, 1 refuted" in md and "Refuted predictions (shown, not hidden)" in md and wrong in md
    assert out["prediction_record"]["hit_rate"] == 0.5


def test_exploratory_runs_cannot_be_the_only_support_for_a_claim(session):
    e = session.run_experiment("baseline", syn.copy_data(), exploratory=True)
    with pytest.raises(UnbackedClaim, match="exploratory"):
        session.write_report({"claims": [{"text": "it works", "evidence": [e["id"]]}]})
    pid = pre(session, "baseline", syn.copy_data(), "summary.clean_accuracy", ">=", 0.9)
    r = session.run_experiment(prereg_id=pid)
    out = session.write_report({"claims": [{"text": "it works", "evidence": [e["id"], r["id"]]}]})
    assert "Exploratory runs" in (session._dir() / out["markdown"]).read_text(encoding="utf-8")


def test_prereg_needs_a_quantitative_prediction_and_an_alternative(session):
    spec = {"primitive": "baseline", "params": syn.copy_data()}
    with pytest.raises(ValueError):
        session.preregister("h", spec, {"metric": "summary.clean_accuracy", "op": "roughly", "threshold": 1}, ["alt"])
    with pytest.raises(ValueError):
        session.preregister("h", spec, {"metric": "summary.clean_accuracy", "op": ">=", "threshold": "high"}, ["alt"])
    with pytest.raises(ValueError):
        session.preregister("h", spec, {"metric": "summary.clean_accuracy", "op": ">=", "threshold": 0.9}, [])
    with pytest.raises(KeyError):
        session.preregister("h", {"primitive": "seance", "params": {}}, {"metric": "summary.x", "op": ">=", "threshold": 1}, ["alt"])


# ---- hash chain --------------------------------------------------------------------------------------------

def _book(tmp_path, n=5):
    nb = Notebook(tmp_path / "notebook.jsonl")
    for i in range(n):
        nb.append("note", f"entry {i}")
    return nb


def _rewrite(nb, fn):
    lines = [json.loads(l) for l in nb.path.read_text(encoding="utf-8").splitlines()]
    nb.path.write_text("\n".join(json.dumps(e) for e in fn(lines)) + "\n", encoding="utf-8")


def test_clean_chain_verifies(tmp_path):
    nb = _book(tmp_path)
    v = nb.verify()
    assert v["ok"] and v["chained"] == 5 and v["first_problem"] is None
    assert nb.entries()[0]["prev"] == "0" * 64 and nb.entries()[1]["prev"] == nb.entries()[0]["hash"]


def test_edit_is_detected_at_the_edited_entry(tmp_path):
    nb = _book(tmp_path)

    def edit(lines):
        lines[2]["text"] = "the prediction was always this"
        return lines

    _rewrite(nb, edit)
    v = nb.verify()
    assert not v["ok"] and v["first_problem"]["id"] == "N0003" and "edited" in v["first_problem"]["reason"]


def test_reorder_and_deletion_are_detected(tmp_path):
    nb = _book(tmp_path)
    _rewrite(nb, lambda l: [l[0], l[2], l[1], l[3], l[4]])
    assert nb.verify()["first_problem"]["index"] == 1
    nb2 = _book(tmp_path / "b")
    _rewrite(nb2, lambda l: [l[0], l[1], l[3], l[4]])
    assert nb2.verify()["first_problem"]["index"] == 2


def test_recomputing_one_hash_still_breaks_the_next_link(tmp_path):
    from glassbox.notebook import entry_hash

    nb = _book(tmp_path)

    def forge(lines):
        lines[1]["text"] = "forged"
        lines[1]["hash"] = entry_hash(lines[1]["prev"], lines[1])
        return lines

    _rewrite(nb, forge)
    v = nb.verify()
    assert not v["ok"] and v["first_problem"]["id"] == "N0003" and "N0002" in v["first_problem"]["reason"]


def test_older_unchained_notebooks_can_be_continued(tmp_path):
    path = tmp_path / "notebook.jsonl"
    path.write_text(json.dumps({"id": "N0001", "ts": "2026-11-12T00:00:00", "kind": "note", "text": "old", "refs": []}) + "\n",
                    encoding="utf-8")
    nb = Notebook(path)
    nb.append("note", "new")
    v = nb.verify()
    assert v["ok"] and v["legacy_unchained_prefix"] == 1 and v["chained"] == 1
    _rewrite(nb, lambda l: [{**l[0], "text": "old, edited"}, l[1]])
    assert not nb.verify()["ok"]                                            # the old entry is anchored by the first chained one


def test_prereg_entries_are_part_of_the_chain(session):
    pid = pre(session, "baseline", syn.copy_data(), "summary.clean_accuracy", ">=", 0.9)
    session.run_experiment(prereg_id=pid)
    assert session.notebook_verify()["ok"]
    _rewrite(session.notebook, lambda l: [({**e, "prediction": {**e["prediction"], "threshold": 0.1}} if e["id"] == pid else e) for e in l])
    v = session.notebook_verify()
    assert not v["ok"] and v["first_problem"]["id"] == pid
