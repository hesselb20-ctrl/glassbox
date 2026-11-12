"""Every lab primitive against a model whose mechanism is known by construction."""
import json
import os

import pytest
import torch

from glassbox.backends.base import PRIMITIVES, UnsupportedPrimitive, require
from glassbox.lab import REGISTRY, Lab

from . import synthetic as syn


def A(x):
    return pytest.approx(x, abs=1e-4)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    return Lab(syn.backend(syn.copy_model()), tmp_path_factory.mktemp("run"))


@pytest.fixture(scope="module")
def data():
    return syn.copy_data()


def test_registry_matches_capability_table():
    assert set(REGISTRY) == set(PRIMITIVES)


def test_baseline(lab, data):
    r = lab.run("baseline", data)
    assert r["summary"]["clean_metric_mean"] == A(syn.GAIN)
    assert r["summary"]["corrupt_metric_mean"] == A(-syn.GAIN)
    assert r["summary"]["clean_accuracy"] == 1.0
    assert os.path.exists(lab.out_dir / r["plot"])


def test_activation_patching_heads_finds_the_copy_head(lab, data):
    eff = lab.run("activation_patching", {**data, "component": "head"})["data"]["effect"]
    assert eff[0][0] == A(1.0) and eff[0][1] == A(0.0)


def test_activation_patching_resid_finds_position_zero(lab, data):
    eff = lab.run("activation_patching", {**data, "component": "resid_pre"})["data"]["effect"]
    assert eff[0] == [A(1.0), A(0.0), A(0.0), A(0.0)]


def test_noising_direction_is_the_mirror_image(lab, data):
    eff = lab.run("activation_patching", {**data, "component": "head", "direction": "noise"})["data"]["effect"]
    assert eff[0][0] == A(1.0) and eff[0][1] == A(0.0)


def test_attribution_patching_is_exact_on_a_linear_circuit(lab, data):
    r = lab.run("attribution_patching", data)
    assert r["data"]["heads"][0] == [A(1.0), A(0.0)]
    assert r["data"]["resid_pre"][0] == [A(1.0), A(0.0), A(0.0), A(0.0)]
    assert r["summary"]["top_heads"][0]["component"] == "0.0"


def test_path_patching_direct_to_logits(lab, data):
    eff = lab.run("path_patching", {**data, "receiver": "logits"})["data"]["effect"]
    assert eff[0] == [A(1.0), A(0.0)]


@pytest.mark.parametrize("mode", ["zero", "mean", "resample"])
def test_ablation_modes(lab, data, mode):
    r = lab.run("ablation", {**data, "sweep": "heads", "mode": mode})
    by = dict(zip(r["data"]["components"], r["data"]["metric"]))
    assert by["0.1"] == A(syn.GAIN)                # the dead head does not matter
    assert by["0.0"] < syn.GAIN - 1.0              # the copy head does
    if mode == "zero":
        assert by["0.0"] == A(0.0)


def test_joint_ablation(lab, data):
    r = lab.run("ablation", {**data, "components": ["0.0", "0.1"], "mode": "zero", "joint": True})
    assert r["summary"]["metric_ablated"] == A(0.0)


def test_logit_lens_reads_the_answer(lab, data):
    r = lab.run("logit_lens", data)
    assert r["summary"]["metric_by_layer"] == [A(syn.GAIN)]
    assert r["summary"]["median_rank_of_answer"] == [0.0]


def test_direct_logit_attribution(lab, data):
    c = lab.run("direct_logit_attribution", data)["data"]["contribution"]
    assert c[0] == [A(syn.GAIN), A(0.0)]


def test_attention_patterns(lab, data):
    r = lab.run("attention_patterns", {**data, "query_pos": -1, "key_pos": 0})
    assert r["data"]["score"][0][0] == A(1.0)       # copy head: all attention on position 0
    assert r["data"]["score"][0][1] == A(0.25)      # dead head: uniform over 4 positions
    r = lab.run("attention_patterns", {**data, "head": "0.0"})
    assert len(r["data"]["pattern"]) == 4


def _two_class(n=20):
    g = torch.Generator().manual_seed(1)

    def rows(t):
        return [[t] + torch.randint(0, syn.V, (3,), generator=g).tolist() for _ in range(n)]

    return rows(1), rows(2)


def test_diff_of_means_and_steering(lab):
    pos, neg = _two_class()
    r = lab.run("diff_of_means", {"positive": {"tokens": pos}, "negative": {"tokens": neg}})
    assert r["summary"]["heldout_accuracy"] == [1.0]
    art = r["summary"]["direction_artifact"]
    d = lab.direction({"artifact": art}, "resid_post.0")
    assert d[syn.OUT0 + 1] == A(1.0) and d[syn.OUT0 + 2] == A(-1.0)
    # steering along the unit direction moves the logit difference linearly, by GAIN times its output-dim part
    # (the direction also carries a little sampling noise in the token dims, which the unembedding ignores)
    expected = syn.GAIN * float(d[syn.OUT0 + 1] - d[syn.OUT0 + 2]) / float(d.norm())
    s = lab.run("steering", {"direction": {"artifact": art}, "sites": ["resid_post.0"], "alphas": [-2, -1, 0, 1, 2],
                             "scale": "absolute", "clean": {"tokens": neg},
                             "metric": {"type": "logit_diff", "correct": [1], "incorrect": [2]}})
    assert s["summary"]["slope_metric_per_alpha"] == A(expected)
    assert expected > 0.95 * syn.GAIN * 2 ** 0.5
    assert s["summary"]["metric_by_alpha"]["0.0"] == A(-syn.GAIN)


def test_linear_probe(lab):
    pos, neg = _two_class(30)
    r = lab.run("linear_probe", {"data": {"tokens": pos + neg}, "labels": ["one"] * 30 + ["two"] * 30})
    assert r["summary"]["best_accuracy"] == 1.0
    assert r["summary"]["shuffled_label_control"][0] < 0.8


def test_periodicity_recovers_planted_frequencies(tmp_path):
    lab = Lab(syn.backend(syn.fourier_model()), tmp_path)
    r = lab.run("periodicity", {"period": 16, "source": "W_E"})
    assert r["summary"]["key_frequencies"] == [3, 5]
    assert r["summary"]["share_of_power_in_key_frequencies"] == A(1.0)
    art = lab.load_artifact(r["summary"]["subspace_artifact"])
    assert art["basis"].shape == (4, 4)


def test_subspace_ablation_removes_exactly_one_frequency(tmp_path):
    b = syn.backend(syn.fourier_model())
    lab = Lab(b, tmp_path)
    aid = lab.run("periodicity", {"period": 16, "source": "W_E"})["summary"]["subspace_artifact"]
    toks = torch.tensor([[a, 16] for a in range(16)])
    edit = lab.subspace_edit({"artifact": aid, "op": "project_out", "groups": [3], "pos": [0]})
    out = b.forward(toks, read=["embed"], edits=[edit]).acts["embed"][:, 0]
    assert out[:, :2].abs().max() < 1e-4            # frequency 3 is gone
    assert out[:, 2:].abs().max() > 0.9             # frequency 5 is untouched


def test_weights_svd(tmp_path):
    lab = Lab(syn.backend(syn.fourier_model()), tmp_path)
    r = lab.run("weights_svd", {"weight": "W_E", "rows": 16})
    assert r["summary"]["dims_for_99pct_variance"] == 4


def test_unsupported_primitive_names_a_backend_that_works():
    with pytest.raises(UnsupportedPrimitive) as e:
        require("local-gguf", "activation_patching")
    assert "local-full" in str(e.value) and "ndif" in str(e.value)


def test_records_are_written_and_json_clean(lab, data):
    r = lab.run("baseline", data)
    on_disk = lab.load_experiment(r["id"])
    assert on_disk["summary"] == json.loads(json.dumps(r["summary"]))
    assert on_disk["provenance"]["backend"] == "local-full"
    assert on_disk["provenance"]["counts_as_verification"] is True
