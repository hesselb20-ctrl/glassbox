"""Run a demo investigation headless, the same way the benchmarks are run.

    python demos/run_demo.py qwen27b-language

Uses the local-gguf backend on a GGUF file from ``GLASSBOX_GGUF_DIR`` (default ``~/models``) through the
llama.cpp build in ``GLASSBOX_LLAMA_DIR`` (default ``~/llama.cpp``). Zero download. Results go to demos/<name>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from run_blind import main  # noqa: E402

DEMOS = {
    "qwen27b-language": dict(
        backend="local-gguf", model="Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf", budget=16,
        behavior="This is a 27B model, a quantized (about 4.8 bits per weight) and abliterated community variant of "
                 "Qwen3.8-27B, running through llama.cpp on a home PC. Behaviour: it continues a text in the language "
                 "the text is written in (an English sentence beginning gets an English continuation, a French one a "
                 "French continuation). Where in the network is the language of the continuation represented, and is "
                 "there a direction for it? This backend can read activations and steer, but it cannot patch. Write "
                 "your own prompts (at least 30 short sentence beginnings per language, different topics, plain text, "
                 "no chat template). The model is large, so keep every experiment under "
                 "about 80 prompts. Find the layer range, extract a direction, then test it causally: steering English "
                 "prompts along it should give French continuations, and steering French prompts against it should "
                 "give English ones. Include generated text before and after steering. Everything found here is a "
                 "screening result on a quantized, abliterated variant, and must be labelled that way.",
    ),
    # An open case: a behaviour with no published circuit that we know of for this model. Whatever comes out is unverified
    # by any outside ground truth and is reported as such.
    "gpt2-a-an": dict(
        home="open_cases", backend="local-full", model="gpt2", budget=40,
        behavior="GPT-2 small chooses between the articles ' a' and ' an' according to the word that will come next, a word "
                 "it has not produced yet. Example: 'I climbed up the pear tree and picked a pear. I climbed up the apple "
                 "tree and picked' is continued with ' an'. How does it decide? There is no built-in dataset task for this: "
                 "write your own position-aligned clean and corrupt prompt pairs (the same number of tokens in each pair, "
                 "check it with the baseline), and use the logit difference between ' an' and ' a' as the metric.",
    ),
    "gpt2-agreement": dict(
        home="open_cases", backend="local-full", model="gpt2", budget=40,
        behavior="GPT-2 small makes the verb agree with the subject even when another noun sits in between: "
                 "'The keys to the cabinet' is continued with ' are' rather than ' is', and 'The key to the cabinets' with "
                 "' is'. How does it keep track of the number of the subject across the distractor noun? There is no "
                 "built-in dataset task for this: write your own position-aligned clean and corrupt prompt pairs (the same "
                 "number of tokens in each pair; the corrupt prompt flips the number of the subject), check them with the "
                 "baseline, and use the logit difference between the correct and the incorrect verb form as the metric.",
    ),
}

if __name__ == "__main__":
    name = sys.argv[1]
    spec = dict(DEMOS[name])
    sys.exit(main(name, spec, home=spec.pop("home", "demos")))
