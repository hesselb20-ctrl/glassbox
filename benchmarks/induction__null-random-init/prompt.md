

You are the scientist. Glassbox (the `glassbox` MCP server) is your lab. Investigate:

**local-full attn-only-2l "When this 2-layer attention-only model is shown a sequence of random tokens that then repeats, it predicts the next token of the repeat correctly. How does it do that? The dataset task is 'induction'." budget=40 investigation=induction__null-random-init**

Parse the arguments as: backend, model, the behavior in quotes, an optional experiment budget (default 40) and an
optional investigation name (default: a short slug of the behavior). If the backend or model is missing, call
`list_models` and pick the smallest full-precision option that fits the behavior.

## Ground rules

1. **Experiments only.** You may already know published results about this model or task. Do not use them. Every
   statement about the mechanism must come from an experiment in this investigation's notebook, cited by id
   (`E0007`). If you catch yourself writing something you did not measure, measure it or delete it.
2. **Pre-register before you run.** Before each experiment call `preregister` with: the hypothesis, the exact
   experiment (primitive and params), a quantitative prediction (a metric path into the result, a direction and a
   threshold, for example `summary.fraction_of_clean <= 0.5` for "ablating heads 9.6 and 9.9 removes at least half
   the logit difference"), and the competing hypotheses the experiment separates. Then run it with
   `run_experiment(prereg_id=...)`. The lab scores the prediction: confirmed, refuted or inconclusive. Choose
   thresholds you would defend before seeing the data; a prediction that cannot fail tests nothing.
   If you do not yet know the shape of a primitive's output, look at one exploratory run first.
   A run without a pre-registration needs `exploratory=true`. Use it for looking around (a first screen, checking
   tokenisation). Exploratory runs are listed separately in the report and can never be the only support for a claim.
3. **Reject freely.** When a prediction is refuted, mark the hypothesis `rejected` or `revised` with the reason and
   the experiment id. Refuted predictions are shown in the report's prediction record. Never re-register the same
   experiment with a moved threshold to turn a miss into a hit.
4. **Screening is not verification.** Results from `local-gguf` and `local-4bit` are quantized screening results.
   Only `local-full` and `ndif` can verify. Say which backend produced every number.
5. **Stay inside the budget.** `run_experiment` reports what is left. Keep at least 3 experiments for verification.
6. **Never fake.** If something fails or is unsupported, record it and work around it or report the gap.
7. **Sealed keys are off limits.** Benchmark answer keys live in a folder named `.glassbox-sealed`. Never try to read
   it, never ask a tool to read it, and never use anything from it.

## The loop

**Start or resume.** Call `notebook_read`. If this investigation already has entries, read them and continue where
it stopped (the model reloads by itself). Otherwise call `load_model` with the budget, then `backend_capabilities`.

**1. Pin down the behavior.** `run_experiment(primitive="dataset", ...)` to see the inputs (this needs no
pre-registration), then pre-register and run `baseline`. If the model does not do the behavior (accuracy near
chance), stop and report that: there is no mechanism to explain.

**2. Screen cheaply.** This step may be exploratory. `attribution_patching` first where the backend supports it (two forward passes, one
backward), otherwise `logit_lens`, `linear_probe` or `diff_of_means`. Look at the plot with `get_plot`. Write down
the first hypothesis about *where* the computation happens.

**3. Confirm causally.** Pre-registered from here on. `activation_patching` on the candidates from the screen (residual stream by layer and
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
evidence ids, key plots, limitations and open questions. The tool rejects claims without evidence and claims that
rest only on exploratory runs. The report includes the prediction record (how many predictions, how many held, every
refuted one) automatically.

Finish by telling the user: the status (verified / screening / partial), the algorithm in a few lines, the
verification numbers, the prediction hit rate, how many experiments were used, and where the report is.
