"""Null controls: null models, shuffled labels, null-calibrated verification, false discovery counting."""
import json

import pytest
import torch

from glassbox import nulls
from glassbox.lab import Lab
from glassbox.session import Session
from glassbox.verify import null_test

from . import synthetic as syn

COPY_CODE = "def predict(tokens, prompt):\n    return tokens[0]\n"


@pytest.fixture()
def session(tmp_path):
    s = Session(tmp_path)
    s._open("t")
    s.backend = syn.backend(syn.copy_model())
    s.lab = Lab(s.backend, s._dir())
    s._save_state(backend="local-full", model="synthetic", budget=50)
    return s


def test_null_test_passes_only_above_the_95th_percentile():
    null = [float(i) for i in range(100)]            # 0..99: 95th percentile is 94.05
    assert null_test("x", 99.5, null, {"k": 100}, 95.0, "")["pass"] is True
    assert null_test("x", 94.0, null, {"k": 100}, 95.0, "")["pass"] is False
    r = null_test("x", 50.0, null, {"k": 100}, 95.0, "")
    assert r["n_null"] == 100 and r["null_mean"] == pytest.approx(49.5) and r["percentile_of_effect"] == 50.0
    assert null_test("x", 1.0, [], {}, 95.0, "")["pass"] is False          # no nulls, no pass


def test_weight_shuffle_keeps_statistics_and_destroys_the_behaviour():
    b = syn.backend(syn.copy_model())
    lab = Lab(b, __import__("tempfile").mkdtemp())
    data = syn.copy_data(24)
    assert lab.run("baseline", data)["summary"]["clean_accuracy"] == 1.0
    nb = b.null_copy("weight-shuffled", 0)
    for (n1, p1), (_, p2) in zip(b.model.named_parameters(), nb.model.named_parameters()):
        assert torch.allclose(p1.flatten().sort().values, p2.flatten().sort().values), n1     # same values, new places
    assert Lab(nb, __import__("tempfile").mkdtemp()).run("baseline", data)["summary"]["clean_accuracy"] < 0.5
    assert lab.run("baseline", data)["summary"]["clean_accuracy"] == 1.0                      # the original is untouched


def test_random_init_null_has_the_same_architecture():
    b = syn.backend(syn.copy_model())
    nb = b.null_copy("random-init", 1)
    assert {k: tuple(v.shape) for k, v in b.model.state_dict().items()} == {k: tuple(v.shape) for k, v in nb.model.state_dict().items()}
    assert not torch.allclose(b.model.embed.W_E, nb.model.embed.W_E)


def test_task_shuffled_null_removes_the_behaviour_without_touching_the_model(session):
    data = syn.copy_data(40)
    assert session.run_experiment("baseline", data, exploratory=True)["summary"]["clean_accuracy"] == 1.0
    session.lab.label_shuffle_seed = 0
    assert session.run_experiment("baseline", data, exploratory=True)["summary"]["clean_accuracy"] < 0.5


def test_verification_reports_null_calibration_for_every_finding(session):
    v = session.verify_hypothesis({"name": "copy", **syn.copy_data(40, seed=9), "code": COPY_CODE,
                                   "circuit": {"components": ["0.0"], "mode": "zero"}})
    assert v["status"] == "verified" and v["beats_null_controls"] is True
    names = {n["finding"] for n in v["null_controls"]}
    assert names == {"agreement", "necessity", "sufficiency"}
    for n in v["null_controls"]:
        assert n["pass"] and n["n_null"] > 0 and n["effect"] > n["null_p95"] and "null_mean" in n
    nec = next(n for n in v["null_controls"] if n["finding"] == "necessity")
    assert {"random-same-size-set", "random-init", "weight-shuffled"} <= set(nec["null_samples_by_kind"])


def test_a_circuit_no_better_than_random_fails_the_null_check(session):
    """Head 0.1 does nothing. Lower the thresholds until they pass: the null comparison must still reject it."""
    v = session.verify_hypothesis({"name": "dead head", **syn.copy_data(40, seed=9),
                                   "circuit": {"components": ["0.1"], "mode": "zero"},
                                   "thresholds": {"circuit_alone_min": -10, "without_circuit_max": 10}})
    assert v["all_checks_passed"] is True                       # the (absurd) thresholds are met ...
    assert v["status"] == "not-verified" and v["beats_null_controls"] is False      # ... but it does not beat the nulls
    assert "null distribution" in v["note"]


def test_verification_under_the_task_shuffled_null_finds_nothing(session):
    session.lab.label_shuffle_seed = 3
    v = session.verify_hypothesis({"name": "copy on shuffled labels", **syn.copy_data(40, seed=9), "code": COPY_CODE,
                                   "circuit": {"components": ["0.0"], "mode": "zero"}})
    assert v["status"] == "not-verified"


def test_null_is_recorded_on_disk_but_not_shown_to_the_scientist(session):
    session._save_state(null={"kind": "task-shuffled", "seed": 0})
    session.lab.label_shuffle_seed = 0
    r = session.run_experiment("baseline", syn.copy_data(), exploratory=True)
    assert "null" not in json.dumps(r)
    assert session.lab.load_experiment(r["id"])["null"] == {"kind": "task-shuffled", "seed": 0}


def test_parse_and_env(monkeypatch):
    assert nulls.parse("weight-shuffled:3") == {"kind": "weight-shuffled", "seed": 3}
    assert nulls.parse(None) is None
    with pytest.raises(ValueError):
        nulls.parse("telepathy")
    monkeypatch.setenv("GLASSBOX_NULL", "random-init:1")
    assert nulls.from_env() == {"kind": "random-init", "seed": 1}


def test_false_discovery_rate_counts_null_runs_that_report_a_circuit(tmp_path):
    def run(name, null, verif=None, report=None):
        d = tmp_path / name
        (d / "experiments").mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"null": null}))
        (d / "experiments" / "E0001_baseline.json").write_text("{}")
        if verif:
            (d / "experiments" / "V0001_verify.json").write_text(json.dumps({"status": verif}))
        if report:
            (d / "report").mkdir()
            (d / "report" / "report.json").write_text(json.dumps({"overall_status": report}))

    run("ioi__null-random-init", {"kind": "random-init", "seed": 0})
    run("ioi__null-weight-shuffled", {"kind": "weight-shuffled", "seed": 0}, verif="not-verified", report="partial / not verified")
    run("ioi__null-task-shuffled", {"kind": "task-shuffled", "seed": 0}, verif="verified", report="verified")
    run("other__null-random-init", {"kind": "random-init", "seed": 0}, verif="verified")
    out = nulls.summarize_null_runs(tmp_path, "ioi")
    assert out["n"] == 3 and out["false_discoveries"] == 1 and out["false_discovery_rate"] == pytest.approx(1 / 3)
