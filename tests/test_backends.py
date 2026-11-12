"""Every backend returns the same JSON shape for the same primitive, and refuses what it cannot do.

Real hardware is not needed: local-full and the two nnsight backends run a tiny random GPT-2 on CPU
(ndif is exercised with remote execution switched off, so the code path is the same but nothing leaves
the machine), and local-gguf runs against a stand-in for the llama.cpp worker process.
"""
import numpy as np
import pytest
import torch

from glassbox.backends import BACKENDS, Edit, UnsupportedPrimitive, capability_table, get_backend
from glassbox.backends.base import PRIMITIVES
from glassbox.lab import Lab

D, LAYERS, VOCAB = 16, 2, 50257
RECORD_KEYS = {"id", "primitive", "provenance", "params", "summary", "data", "plot", "seconds", "timestamp"}
PROVENANCE_KEYS = {"backend", "precision", "tag", "model", "counts_as_verification"}


class FakeLlamaWorker:
    """Stands in for the llama.cpp worker: deterministic activations with the right names and shapes."""

    def __init__(self) -> None:
        self.control = None

    def call(self, cmd, **a):
        if cmd == "load":
            return {"n_embd": D, "n_layer": LAYERS, "n_head": 2, "n_vocab": VOCAB, "arch": "fake", "name": "fake",
                    "expert_count": None, "rms_eps": "1e-6", "file_type": "15"}
        if cmd == "tokenize":
            return [[(ord(c) * 7) % VOCAB for c in t][:8] or [1] for t in a["texts"]]
        if cmd == "pieces":
            return [f"<{t}>" for t in a["tokens"]]
        if cmd == "names":
            return {"l_out-0": (D, 2), "attn_residual-0": (D, 2), "ffn_out-0": (D, 2), "model.input_embed": (D, 2)}
        if cmd == "run":
            toks = a["tokens"]
            g = np.random.default_rng(sum(toks))
            got = {n: g.standard_normal((len(toks), D)).astype(np.float32) + (toks[0] % 2) for n in a["want"]}
            for layer, v in (a.get("control") or {}).items():
                if f"l_out-{layer}" in got:
                    got[f"l_out-{layer}"] += v
            logits = g.standard_normal(VOCAB).astype(np.float32)
            if a.get("control"):
                logits[:D] += sum(a["control"].values())
            return logits, got
        if cmd == "generate":
            return [1, 2, 3]
        return None


@pytest.fixture(scope="module")
def tiny_hf():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
    except Exception as e:  # offline
        pytest.skip(f"needs the GPT-2 tokenizer from the Hugging Face cache or network: {e}")
    cfg = transformers.GPT2Config(n_layer=LAYERS, n_head=2, n_embd=D, vocab_size=VOCAB, n_positions=32)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg).eval(), tok


@pytest.fixture(scope="module")
def labs(tiny_hf, tmp_path_factory):
    from nnsight.modeling.transformers import TransformersModel
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    hf, tok = tiny_hf
    out = {}

    cfg = HookedTransformerConfig(n_layers=LAYERS, n_heads=2, d_model=D, d_head=8, d_mlp=32, act_fn="gelu", n_ctx=32,
                                  d_vocab=VOCAB, tokenizer_name="gpt2", seed=0)
    full = get_backend("local-full", device="cpu")
    full.load(model=HookedTransformer(cfg), model_id="tiny")
    out["local-full"] = full

    for name in ("local-4bit", "ndif"):
        b = get_backend(name)
        b.remote = False
        b.attach(TransformersModel(hf, tokenizer=tok, task="text-generation"), "tiny")
        out[name] = b

    gguf = get_backend("local-gguf")
    gguf.worker = FakeLlamaWorker()
    gguf.meta = gguf.worker.call("load")
    gguf.model_id = "fake-Q4_K_M.gguf"
    gguf.names = {"resid_post": "l_out", "resid_mid": "attn_residual", "mlp_out": "ffn_out", "embed": "model.input_embed"}
    gguf._unembed = (torch.ones(D), torch.randn(VOCAB, D))
    out["local-gguf"] = gguf
    return {k: Lab(v, tmp_path_factory.mktemp(k.replace("-", "_"))) for k, v in out.items()}


POS = ["I love this", "This is great", "What a nice day", "So happy now", "Truly wonderful stuff", "A lovely time"]
NEG = ["I hate this", "This is awful", "What a bad day", "So sad now", "Truly terrible stuff", "A dreadful time"]


def _shape(x):
    """Reduce a JSON value to its structure: key sets and types, not values."""
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return ["list", _shape(x[0]) if x and isinstance(x[0], (dict, list)) else type(x[0]).__name__ if x else None]
    return "number" if isinstance(x, (int, float)) and not isinstance(x, bool) else type(x).__name__


CALLS = [
    ("diff_of_means", lambda: {"positive": {"prompts": POS}, "negative": {"prompts": NEG}, "sites": ["resid_post.1"]}),
    ("linear_probe", lambda: {"data": {"prompts": POS + NEG}, "labels": ["pos"] * 6 + ["neg"] * 6, "sites": ["resid_post.0", "resid_post.1"]}),
    ("logit_lens", lambda: {"clean": {"prompts": POS[:3]}, "metric": {"type": "logit_diff", "correct": [5], "incorrect": [6]}}),
    ("baseline", lambda: {"clean": {"prompts": POS[:3]}, "metric": {"type": "logit_diff", "correct": [5], "incorrect": [6]}}),
]


@pytest.mark.parametrize("primitive,params", CALLS, ids=[c[0] for c in CALLS])
def test_same_json_shape_on_every_backend(labs, primitive, params):
    recs = {name: lab.run(primitive, params()) for name, lab in labs.items()}
    assert set(recs) == set(BACKENDS)
    ref = recs["local-full"]
    for name, r in recs.items():
        assert set(r) == RECORD_KEYS, name
        assert set(r["provenance"]) >= PROVENANCE_KEYS, name
        assert r["provenance"]["backend"] == name
        assert r["provenance"]["tag"].startswith(BACKENDS[name]["tag"].rstrip(")")), name
        assert set(r["summary"]) == set(ref["summary"]), f"{name} summary keys differ from local-full"
        assert _shape(r["data"]) == _shape(ref["data"]), f"{name} data shape differs from local-full"
        assert r["plot"].endswith(".png")


def test_steering_has_the_same_shape_everywhere(labs):
    recs = {}
    for name, lab in labs.items():
        art = lab.run("diff_of_means", {"positive": {"prompts": POS}, "negative": {"prompts": NEG},
                                        "sites": ["resid_post.1"]})["summary"]["direction_artifact"]
        recs[name] = lab.run("steering", {"direction": {"artifact": art}, "sites": ["resid_post.1"], "alphas": [-1, 0, 1],
                                          "clean": {"prompts": POS[:2]},
                                          "metric": {"type": "logit_diff", "correct": [5], "incorrect": [6]}})
    ref = recs["local-full"]
    for name, r in recs.items():
        assert set(r["summary"]) == set(ref["summary"]), name
        assert len(r["data"]["metric"]) == 3


def test_only_full_precision_backends_count_as_verification(labs):
    verdict = {name: lab.backend.provenance()["counts_as_verification"] for name, lab in labs.items()}
    assert verdict == {"local-full": True, "ndif": True, "local-4bit": False, "local-gguf": False}
    assert "screening" in labs["local-gguf"].backend.tag and "screening" in labs["local-4bit"].backend.tag


def test_gguf_refuses_patching_and_names_backends_that_can(labs):
    lab = labs["local-gguf"]
    with pytest.raises(UnsupportedPrimitive) as e:
        lab.run("activation_patching", {"clean": {"prompts": POS[:2]}, "corrupt": {"prompts": NEG[:2]},
                                        "metric": {"type": "logit_diff", "correct": [5], "incorrect": [6]}})
    assert "local-full" in str(e.value) and "ndif" in str(e.value) and "local-4bit" in str(e.value)
    with pytest.raises(UnsupportedPrimitive):
        lab.backend.forward(torch.tensor([[1, 2, 3]]), edits=[Edit("resid_post.1", "replace", torch.zeros(D))])


def test_nnsight_backend_matches_local_patching_semantics(labs):
    """Zero-ablating a head on the nnsight path changes the logits; the same edit twice gives the same answer."""
    b = labs["local-4bit"].backend
    toks = b.encode(POS[:2])
    clean = b.forward(toks).logits
    e = [Edit("head_out.1", "zero", None, None, 0)]
    a1, a2 = b.forward(toks, edits=e).logits, b.forward(toks, edits=e).logits
    assert torch.allclose(a1, a2) and not torch.allclose(a1, clean)
    assert torch.allclose(b.forward(toks).logits, clean)        # edits do not leak into the next run


def test_capability_table_covers_every_primitive_and_backend():
    t = capability_table()
    assert set(t["primitives"]) == set(PRIMITIVES)
    for row in t["primitives"].values():
        assert set(row) == set(BACKENDS)
    assert t["primitives"]["activation_patching"] == {"local-full": True, "local-gguf": False, "local-4bit": True, "ndif": True}
    assert t["primitives"]["steering"] == {b: True for b in BACKENDS}
