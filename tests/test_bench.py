"""glassbox-bench: obfuscation keeps behaviour, keys stay sealed, grading is exact for a known answer."""
import json
import os
import random

import pytest
import torch

from glassbox import tasks
from glassbox.bench import behavior, truth
from glassbox.bench import generate as G

from . import synthetic as syn


def _copy_state():
    m = syn.copy_model()
    cfg = dict(n_layers=1, d_model=20, d_head=8, n_heads=2, n_ctx=4, d_vocab=syn.V, d_vocab_out=syn.V, attn_only=False,
               d_mlp=4, act_fn="relu", normalization_type=None, seed=0)
    sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if not k.endswith(("mask", "IGNORE"))}
    sd.update({"blocks.0.mlp.W_in": torch.zeros(20, 4), "blocks.0.mlp.b_in": torch.zeros(4),
               "blocks.0.mlp.W_out": torch.zeros(4, 20), "blocks.0.mlp.b_out": torch.zeros(20)})
    return cfg, sd


def test_obfuscation_preserves_every_output_and_tracks_the_true_head():
    cfg, sd = _copy_state()
    plain = G.build_model(cfg, sd)
    xs = torch.randint(0, syn.V, (200, 4), generator=torch.Generator().manual_seed(0))
    ref = plain(xs)[:, -1]
    assert G.true_components(sd, 1) == ["0.0"]                     # the dead head and the empty MLP are not components
    for seed in range(3):
        cfg2, sd2, key = G.obfuscate(cfg, sd, random.Random(seed))
        model = G.build_model(cfg2, sd2)
        assert cfg2["d_model"] > cfg["d_model"] and cfg2["n_heads"] > cfg["n_heads"] and cfg2["dtype"] == "float64"
        assert torch.allclose(model(xs)[:, -1].float(), ref, atol=1e-6)
        assert (model(xs)[:, -1].argmax(-1) == ref.argmax(-1)).all()
        (head,) = key["true_components"]
        layer, h = map(int, head.split("."))
        # the head the key points at is the one whose removal breaks the model
        broken = {k: v.clone() for k, v in sd2.items()}
        broken[f"blocks.{layer}.attn.W_O"][h] = 0
        assert (G.build_model(cfg2, broken)(xs)[:, -1].argmax(-1) == ref.argmax(-1)).float().mean() < 0.5
        # the rotated weights no longer line up with the original axes
        assert (sd2["embed.W_E"].abs() > 1e-6).float().mean() > 0.9


def test_truth_functions():
    assert truth.hist({"classes": {"a": 0, "b": 1, "c": 0}}, ["a", "b", "c", "a"]) == "3"
    assert truth.sort_by_key({"key": {"a": 2, "b": 5, "c": 5}}, ["b", "a", "c"]) == "c"          # tie: the later one
    assert truth.count_compare({"a": "x", "b": "y", "out": {"gt": "G", "eq": "E", "lt": "L"}}, ["x", "y", "x"]) == "G"
    assert truth.shift_copy({"k": 2, "f": {"a": "b", "b": "c", "c": "a"}}, ["a", "b", "c", "a"]) == "c"
    assert truth.lookup2({"ptr": {"a": 1, "b": 0}, "table": {"b|a": "Z", "a|b": "Y"}}, ["a", "b", "a"]) == "Z"
    assert set(truth.COMPILED) | set(truth.TRAINED) == set(truth.FAMILIES)


def test_the_dataset_task_refuses_the_sealed_folder(tmp_path):
    with pytest.raises(PermissionError):
        tasks.bench(None, path=str(tmp_path / ".glassbox-sealed" / "1"))


def test_the_mcp_server_code_never_touches_the_sealed_folder():
    import pathlib

    root = pathlib.Path(G.__file__).resolve().parent.parent
    readers = [p.name for p in root.rglob("*.py")
               if "sealed_dir(" in p.read_text(encoding="utf-8") and p.name not in ("generate.py", "grade.py")]
    assert readers == []                                          # only the generator (write) and the grader (read) know the path
    for mod in ("mcp/server.py", "session.py", "tasks.py", "lab/core.py"):
        assert "grade" not in (root / mod).read_text(encoding="utf-8").replace("upgrade", "")


def test_the_prompt_does_not_name_the_function():
    text = behavior({"seq_len": 5, "input_token_ids": [2, 3, 4], "track": "compiled"}, "some/path")
    for word in ("sort", "count", "histogram", "lookup", "shift", "copy"):
        assert word not in text.lower()


@pytest.mark.skipif(os.environ.get("GLASSBOX_SLOW") != "1", reason="needs the Tracr environment and about two minutes; set GLASSBOX_SLOW=1")
def test_generate_and_grade_a_seed_end_to_end(tmp_path, monkeypatch):
    from glassbox.bench.grade import grade

    monkeypatch.setenv("GLASSBOX_SEALED", str(tmp_path / "sealed"))
    out = G.generate(12345, tmp_path / "bench")
    assert len(out["tasks"]) == 7
    assert not list((tmp_path / "bench").rglob("*key*")) and len(list((tmp_path / "sealed" / "12345").glob("t*.json"))) == 7
    public = json.loads((tmp_path / "bench" / "seed-12345" / "tasks" / "t1" / "task.json").read_text())
    assert "family" not in public and "params" not in public
    card = grade(12345, tmp_path / "bench")                       # nothing was run: everything scores zero, nothing crashes
    assert card["summary"]["ran"] == 0 and card["summary"]["compiled_solved"] == 0
