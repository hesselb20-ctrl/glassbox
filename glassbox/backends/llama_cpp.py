"""Minimal ctypes bindings to a llama.cpp shared library, built around the ggml eval callback.

llama.cpp lets a caller register ``cb_eval``: a function the graph scheduler calls for every tensor
it computes. Glassbox uses it to copy named tensors (the residual stream after each layer, ...)
out of the forward pass, the same mechanism as llama.cpp's ``eval-callback`` example. Steering uses
llama.cpp's own control-vector support (``llama_set_adapter_cvec``).

The two parameter structs are passed by value, so their layout must match the library. They are
declared here for the llama.cpp API as of build 12032 (commit dde879bc4, November 2026). If your
build is much older or newer, compare them with ``include/llama.h`` of your build.

Point ``GLASSBOX_LLAMA_DIR`` at the folder that holds ``llama.dll`` / ``libllama.so``.
"""
from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np

c_bool, c_int32, c_uint32, c_float, c_void_p, c_char_p, c_size_t = (
    C.c_bool, C.c_int32, C.c_uint32, C.c_float, C.c_void_p, C.c_char_p, C.c_size_t)


class ModelParams(C.Structure):
    _fields_ = [
        ("devices", c_void_p), ("tensor_buft_overrides", c_void_p),
        ("n_gpu_layers", c_int32), ("split_mode", c_int32), ("load_mode", c_int32), ("lazy_mode", c_int32),
        ("main_gpu", c_int32), ("tensor_split", C.POINTER(c_float)),
        ("progress_callback", c_void_p), ("progress_callback_user_data", c_void_p), ("kv_overrides", c_void_p),
        ("vocab_only", c_bool), ("check_tensors", c_bool), ("use_extra_bufts", c_bool), ("no_host", c_bool),
        ("no_alloc", c_bool), ("load_mtp", c_bool),
    ]


EVAL_CB = C.CFUNCTYPE(c_bool, c_void_p, c_bool, c_void_p)


class ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", c_uint32), ("n_batch", c_uint32), ("n_ubatch", c_uint32), ("n_seq_max", c_uint32),
        ("n_rs_seq", c_uint32), ("n_outputs_max", c_uint32), ("n_outputs_max_per_seq", c_uint32),
        ("n_threads", c_int32), ("n_threads_batch", c_int32),
        ("ctx_type", c_int32), ("rope_scaling_type", c_int32), ("pooling_type", c_int32),
        ("attention_type", c_int32), ("flash_attn_type", c_int32),
        ("rope_freq_base", c_float), ("rope_freq_scale", c_float), ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float), ("yarn_beta_fast", c_float), ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32), ("defrag_thold", c_float),
        ("cb_eval", EVAL_CB), ("cb_eval_user_data", c_void_p),
        ("type_k", c_int32), ("type_v", c_int32), ("moe_cache_size", c_size_t),
        ("abort_callback", c_void_p), ("abort_callback_data", c_void_p),
        ("embeddings", c_bool), ("offload_kqv", c_bool), ("no_perf", c_bool), ("op_offload", c_bool),
        ("swa_full", c_bool), ("kv_unified", c_bool),
        ("samplers", c_void_p), ("n_samplers", c_size_t), ("ctx_other", c_void_p),
    ]


class Batch(C.Structure):
    _fields_ = [("n_tokens", c_int32), ("token", C.POINTER(c_int32)), ("embd", C.POINTER(c_float)),
                ("pos", C.POINTER(c_int32)), ("n_seq_id", C.POINTER(c_int32)), ("seq_id", c_void_p),
                ("logits", C.POINTER(C.c_int8))]


class TensorHead(C.Structure):
    """The leading fields of ``struct ggml_tensor`` (all we read directly)."""
    _fields_ = [("type", c_int32), ("buffer", c_void_p), ("ne", C.c_int64 * 4)]


LOG_CB = C.CFUNCTYPE(None, c_int32, c_char_p, c_void_p)
GGML_F32, GGML_F16 = 0, 1


def default_lib_dir() -> Path:
    return Path(os.environ.get("GLASSBOX_LLAMA_DIR", Path.home() / "llama.cpp"))


class Llama:
    """One loaded GGUF model with a context that can capture named tensors and apply control vectors."""

    def __init__(self, model_path: str | Path, lib_dir: str | Path | None = None, n_ctx: int = 1024,
                 n_gpu_layers: int = -1, quiet: bool = True) -> None:
        d = Path(lib_dir) if lib_dir else default_lib_dir()
        win = sys.platform == "win32"
        names = ("llama.dll", "ggml-base.dll", "ggml.dll") if win else ("libllama.so", "libggml-base.so", "libggml.so")
        if not (d / names[0]).exists():
            raise FileNotFoundError(
                f"no {names[0]} in {d}. Set GLASSBOX_LLAMA_DIR to the folder of your llama.cpp build "
                "(the one that contains llama-server).")
        if win:
            os.add_dll_directory(str(d))
            # The GPU backend (ggml-cuda.dll) is loaded later by ggml itself with a plain LoadLibrary, which looks for
            # its dependencies (cublas, cudart) on PATH, not next to the DLL. Without this the model silently runs on CPU.
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        self.L = L = C.CDLL(str(d / names[0]))
        self.G = G = C.CDLL(str(d / names[1]))
        self.GG = C.CDLL(str(d / names[2]))

        def sig(lib, name, res, *args):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, list(args)
            return fn

        sig(L, "llama_backend_init", None)
        sig(L, "llama_model_default_params", ModelParams)
        sig(L, "llama_context_default_params", ContextParams)
        sig(L, "llama_model_load_from_file", c_void_p, c_char_p, ModelParams)
        sig(L, "llama_init_from_model", c_void_p, c_void_p, ContextParams)
        sig(L, "llama_model_get_vocab", c_void_p, c_void_p)
        for f in ("llama_model_n_embd", "llama_model_n_layer", "llama_model_n_head"):
            sig(L, f, c_int32, c_void_p)
        sig(L, "llama_vocab_n_tokens", c_int32, c_void_p)
        sig(L, "llama_vocab_is_eog", c_bool, c_void_p, c_int32)
        sig(L, "llama_tokenize", c_int32, c_void_p, c_char_p, c_int32, C.POINTER(c_int32), c_int32, c_bool, c_bool)
        sig(L, "llama_token_to_piece", c_int32, c_void_p, c_int32, c_char_p, c_int32, c_int32, c_bool)
        sig(L, "llama_batch_get_one", Batch, C.POINTER(c_int32), c_int32)
        sig(L, "llama_decode", c_int32, c_void_p, Batch)
        sig(L, "llama_get_logits_ith", C.POINTER(c_float), c_void_p, c_int32)
        sig(L, "llama_get_memory", c_void_p, c_void_p)
        sig(L, "llama_memory_clear", None, c_void_p, c_bool)
        sig(L, "llama_set_adapter_cvec", c_int32, c_void_p, C.POINTER(c_float), c_size_t, c_int32, c_int32, c_int32)
        sig(L, "llama_model_meta_val_str", c_int32, c_void_p, c_char_p, c_char_p, c_size_t)
        sig(L, "llama_model_free", None, c_void_p)
        sig(L, "llama_free", None, c_void_p)
        sig(L, "llama_log_set", None, LOG_CB, c_void_p)
        sig(G, "ggml_get_name", c_char_p, c_void_p)
        sig(G, "ggml_nbytes", c_size_t, c_void_p)
        sig(G, "ggml_backend_tensor_get", None, c_void_p, c_void_p, c_size_t, c_size_t)

        self._log_cb = LOG_CB(lambda level, text, user: None)
        if quiet:
            L.llama_log_set(self._log_cb, None)
        L.llama_backend_init()
        # builds with dynamic backends (CUDA, CPU variants as separate libraries) need them registered first
        if hasattr(self.GG, "ggml_backend_load_all_from_path"):
            fn = self.GG.ggml_backend_load_all_from_path
            fn.restype, fn.argtypes = None, [c_char_p]
            fn(str(d).encode())

        mp = L.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        self.model = L.llama_model_load_from_file(str(model_path).encode(), mp)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {model_path}")
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_embd = L.llama_model_n_embd(self.model)
        self.n_layer = L.llama_model_n_layer(self.model)
        self.n_head = L.llama_model_n_head(self.model)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.n_ctx = n_ctx
        sig(L, "llama_supports_gpu_offload", c_bool)
        self.gpu_offload = bool(L.llama_supports_gpu_offload())

        self.want: set[str] = set()          # tensor names to capture on the next decode
        self.seen: dict[str, tuple] = {}     # every tensor name seen -> shape (for discovery)
        self.captured: dict[str, list[np.ndarray]] = {}
        self.discover = False
        self._cb = EVAL_CB(self._on_tensor)
        cp = L.llama_context_default_params()
        cp.n_ctx = cp.n_batch = cp.n_ubatch = n_ctx
        cp.cb_eval = self._cb
        cp.no_perf = True
        self.ctx = L.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("llama.cpp could not create a context (out of VRAM?)")

    # -- the eval callback -------------------------------------------------- #
    def _on_tensor(self, t, ask, user) -> bool:
        try:
            name = (self.G.ggml_get_name(t) or b"").decode("utf-8", "replace")
            if ask:
                if self.discover:
                    head = C.cast(t, C.POINTER(TensorHead)).contents
                    self.seen.setdefault(name, tuple(int(x) for x in head.ne))
                return name in self.want
            head = C.cast(t, C.POINTER(TensorHead)).contents
            n = self.G.ggml_nbytes(t)
            buf = np.empty(n, dtype=np.uint8)
            self.G.ggml_backend_tensor_get(t, buf.ctypes.data_as(c_void_p), 0, n)
            if head.type == GGML_F32:
                arr = buf.view(np.float32)
            elif head.type == GGML_F16:
                arr = buf.view(np.float16).astype(np.float32)
            else:
                return True
            shape = [int(x) for x in head.ne if x > 1] or [1]
            self.captured.setdefault(name, []).append(arr.reshape(shape[::-1]).copy())
        except Exception:  # never let an exception cross into C
            pass
        return True

    # -- tokens ------------------------------------------------------------- #
    def tokenize(self, text: str, add_special: bool = True, parse_special: bool = True) -> list[int]:
        raw = text.encode("utf-8")
        cap = len(raw) + 16
        buf = (c_int32 * cap)()
        n = self.L.llama_tokenize(self.vocab, raw, len(raw), buf, cap, add_special, parse_special)
        if n < 0:
            raise RuntimeError("tokenisation failed")
        return list(buf[:n])

    def piece(self, token: int) -> str:
        buf = C.create_string_buffer(256)
        n = self.L.llama_token_to_piece(self.vocab, int(token), buf, 256, 0, True)
        return buf.raw[:max(n, 0)].decode("utf-8", "replace")

    def meta(self, key: str) -> str | None:
        buf = C.create_string_buffer(512)
        n = self.L.llama_model_meta_val_str(self.model, key.encode(), buf, 512)
        return buf.value.decode("utf-8", "replace") if n >= 0 else None

    # -- running ------------------------------------------------------------ #
    def reset(self) -> None:
        self.L.llama_memory_clear(self.L.llama_get_memory(self.ctx), True)

    def decode(self, tokens: list[int], want: set[str] | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Feed tokens (continuing the current sequence). Returns last-position logits and captured tensors [pos, d]."""
        if len(tokens) > self.n_ctx:
            raise ValueError(f"{len(tokens)} tokens exceed the context of {self.n_ctx}")
        self.want, self.captured = set(want or ()), {}
        arr = (c_int32 * len(tokens))(*tokens)
        rc = self.L.llama_decode(self.ctx, self.L.llama_batch_get_one(arr, len(tokens)))
        self.want = set()
        if rc != 0:
            raise RuntimeError(f"llama_decode failed with code {rc}")
        logits = np.ctypeslib.as_array(self.L.llama_get_logits_ith(self.ctx, -1), shape=(self.n_vocab,)).copy()
        got = {k: np.concatenate([a.reshape(-1, a.shape[-1]) for a in v], axis=0) for k, v in self.captured.items()}
        self.captured = {}
        return logits, got

    def set_control(self, vectors: dict[int, np.ndarray] | None) -> None:
        """Add ``vectors[layer]`` to the residual stream after each given layer (all positions). None clears."""
        if not vectors:
            self.L.llama_set_adapter_cvec(self.ctx, None, 0, 0, 0, 0)
            return
        if min(vectors) < 1:
            raise ValueError("llama.cpp control vectors cannot be applied at layer 0")
        data = np.zeros((self.n_layer, self.n_embd), dtype=np.float32)      # row i holds layer i + 1
        for layer, v in vectors.items():
            data[layer - 1] = np.asarray(v, dtype=np.float32)
        flat = np.ascontiguousarray(data.reshape(-1))
        rc = self.L.llama_set_adapter_cvec(self.ctx, flat.ctypes.data_as(C.POINTER(c_float)), flat.size, self.n_embd,
                                           min(vectors), max(vectors))
        if rc != 0:
            raise RuntimeError(f"llama_set_adapter_cvec failed with code {rc}")

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.L.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.L.llama_model_free(self.model)
            self.model = None


# --------------------------------------------------------------------------- #
# Worker process
# --------------------------------------------------------------------------- #
# llama.cpp and PyTorch each bring their own OpenMP runtime, and loading both into one process
# aborts. So the model lives in a small worker process that imports only numpy and this file, and
# the backend talks to it over a local authenticated pipe.

def _serve(address: tuple[str, int], authkey: bytes) -> None:
    from multiprocessing.connection import Client

    conn = Client(address, authkey=authkey)
    m: Llama | None = None
    while True:
        try:
            cmd, a = conn.recv()
        except EOFError:
            break
        try:
            if cmd == "load":
                m = Llama(**a)
                arch = m.meta("general.architecture")
                out = {"n_embd": m.n_embd, "n_layer": m.n_layer, "n_head": m.n_head, "n_vocab": m.n_vocab,
                       "arch": arch, "name": m.meta("general.name"),
                       "expert_count": m.meta(f"{arch}.expert_count"),
                       "rms_eps": m.meta(f"{arch}.attention.layer_norm_rms_epsilon"),
                       "file_type": m.meta("general.file_type"), "gpu_offload_available": m.gpu_offload}
            elif cmd == "tokenize":
                out = [m.tokenize(t, a.get("add_special", True)) for t in a["texts"]]
            elif cmd == "pieces":
                out = [m.piece(t) for t in a["tokens"]]
            elif cmd == "names":
                m.discover, m.seen = True, {}
                m.reset()
                m.decode(a["tokens"])
                m.discover = False
                out = dict(m.seen)
            elif cmd == "run":
                m.reset()
                m.set_control(a.get("control"))
                logits, got = m.decode(a["tokens"], set(a.get("want") or ()))
                out = (logits, got)
            elif cmd == "generate":
                m.reset()
                m.set_control(a.get("control"))
                logits, _ = m.decode(a["tokens"])
                new: list[int] = []
                for _ in range(a["max_new_tokens"]):
                    t = int(np.argmax(logits))
                    if m.L.llama_vocab_is_eog(m.vocab, t):
                        break
                    new.append(t)
                    logits, _ = m.decode([t])
                out = new
            elif cmd == "close":
                if m is not None:
                    m.close()
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown command {cmd}")
            conn.send(("ok", out))
        except Exception as e:  # noqa: BLE001
            conn.send(("err", f"{type(e).__name__}: {e}"))


class Worker:
    """Parent-side handle to the llama.cpp worker process."""

    def __init__(self) -> None:
        import secrets
        import subprocess
        from multiprocessing.connection import Listener

        key = secrets.token_bytes(32)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        env = {**os.environ, "GLASSBOX_WORKER_KEY": key.hex()}
        self.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker",
                                      str(listener.address[1])], env=env, stdout=subprocess.DEVNULL)
        self.conn = listener.accept()
        listener.close()

    def call(self, cmd: str, **kwargs):
        self.conn.send((cmd, kwargs))
        status, out = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"llama.cpp worker: {out}")
        return out

    def close(self) -> None:
        try:
            self.call("close")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.proc.kill()


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    _serve(("127.0.0.1", int(sys.argv[2])), bytes.fromhex(os.environ.pop("GLASSBOX_WORKER_KEY")))
