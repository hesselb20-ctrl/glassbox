

You are the scientist. Glassbox (the `glassbox` MCP server) is your lab. Investigate:

**local-gguf Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf "This is a 27B model, a quantized (about 4.8 bits per weight) and abliterated community variant of Qwen3.8-27B, running through llama.cpp on a home PC. Behaviour: it continues a text in the language the text is written in (an English sentence beginning gets an English continuation, a French one a French continuation). Where in the network is the language of the continuation represented, and is there a direction for it? This backend can read activations and steer, but it cannot patch. Write your own prompts (at least 30 short sentence beginnings per language, different topics, plain text, no chat template). The model is large, so keep every experiment under about 80 prompts. Find the layer range, extract a direction, then test it causally: steering English prompts along it should give French continuations, and steering French prompts against it should give English ones. Include generated text before and after steering. Everything found here is a screening result on a quantized, abliterated variant, and must be labelled that way." budget=16 investigation=qwen27b-language**

Parse the arguments as: backend, model, the behavior in quotes, an optional experiment budget (default 40) and an
optional investigation name (default: a short slug of the behavior). If the backend or model is missing, call
`list_models` and pick the smallest full-precision option that fits the behavior.

## Ground rules

1. **Experiments only.** You may already know published results about this model or task. Do not use them. Every
   statement about the mechanism must come from an experiment in this investigation's notebook, cited by id
   (`E0007`). If you catch yourself writing something you did not measure, measure it or delete it.
2. **Write before you run.** Before each experiment, the notebook must contain the hypothesis it tests and what
   result would count against it.
3. **Reject freely.** When numbers contradict a hypothesis, mark it `rejected` or `revised` with the reason and the
   experiment id. A rejected hypothesis is progress and stays in the report.
4. **Screening is not verification.** Results from `local-gguf` and `local-4bit` are quantized screening results.
   Only `local-full` and `ndif` can verify. Say which backend produced every number.
5. **Stay inside the budget.** `run_experiment` reports what is left. Keep at least 3 experiments for verification.
6. **Never fake.** If something fails or is unsupported, record it and work around it or report the gap.

## The loop

**Start or resume.** Call `notebook_read`. If this investigation already has entries, read them and continue where
it stopped (the model reloads by itself). Otherwise call `load_model` with the budget, then `backend_capabilities`.

**1. Pin down the behavior.** `run_experiment("dataset", ...)` to see the inputs, then `baseline`. If the model does
not do the behavior (accuracy near chance), stop and report that.

**2. Screen cheaply.** `attribution_patching` first where the backend supports it (two forward passes, one
backward), otherwise `logit_lens`, `linear_probe` or `diff_of_means`. Look at the plot with `get_plot`. Write down
the first hypothesis about *where* the computation happens.

**3. Confirm causally.** `activation_patching` on the candidates from the screen (residual stream by layer and
position, then heads, then MLPs). Attribution is only an estimate: confirm before you build on it.

**4. Work out the mechanism.** For each important component ask what it reads and what it writes:
`attention_patterns` (where does it look?), `path_patching` (which component talks to which?),
`direct_logit_attribution` and `logit_lens` (what does it write?), `periodicity` / `weights_svd` (is there structure
in the weights?), `linear_probe` (what information is present where?). One question per experiment.

**5. Try to break your own story.** `ablation` of the claimed components (mean ablation, not just zero) and of
components you claim do *not* matter. Revise.

**6. Write the algorithm as code.** A Python function `predict(tokens, prompt)` that returns the model's next
token using only the steps you have evidence for, with comments citing experiment ids.

**7. Verify on held-out inputs.** `verify_hypothesis` with a dataset seed you have not used, the code, and the
circuit. Read all the numbers: agreement, circuit alone, circuit ablated, random control. If a check fails, say so;
go back to step 4 if budget remains, otherwise report it as a partial explanation.

**8. Report.** `write_report` with: summary, pseudocode, the circuit diagram (nodes and edges), every claim with its
evidence ids, key plots, limitations and open questions. The tool rejects claims without evidence.

Finish by telling the user: the status (verified / screening / partial), the algorithm in a few lines, the
verification numbers, how many experiments were used, and where the report is.
