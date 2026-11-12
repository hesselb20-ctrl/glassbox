# Benchmark scorecard

Each benchmark was run by a fresh headless Claude Code session that received only the /investigate prompt, a one-paragraph description of the behaviour and the glassbox MCP tools: no file access, no web, no shell, no hints about the mechanism. Blind here means exactly that and no more. The scientist is Claude, which has very likely seen these published results in training, so this is not evidence of discovery from zero knowledge. What it does show is that every claim in each report is backed by an experiment that ran in that session (the report tool refuses claims without evidence ids), and how many experiments and minutes that took. This file with the ground truth was written after the three runs had finished.

| Benchmark | Status reported | Rubric: found / partial / missed | Experiments | Wall clock | Hypotheses (rejected or revised) |
|---|---|---|---|---|---|
| Modular addition (1-layer transformer, mod 113) | verified · verified (local-full, full precision) | 3 / 2 / 0 of 5 | 16 (+1 verification) | 4.6 min | 14 (2) |
| Induction (2-layer attention-only model) | partial / not verified · partial: causal checks pass at full precision, agreement check fails | 4 / 1 / 0 of 5 | 17 (+2 verification) | 8.2 min | 12 (2) |
| Indirect object identification (GPT-2 small) | partial / not verified · verified (local-full, full precision) | 4 / 2 / 0 of 6 | 30 (+2 verification) | 10.1 min | 25 (13) |

## Modular addition (1-layer transformer, mod 113)

Ground truth: Nanda, Chan, Lieberum, Smith, Steinhardt (2023), Progress measures for grokking via mechanistic interpretability. The model here was trained from scratch for this repo (benchmarks/modadd/model/training.json), so the key frequencies are specific to this training run.

Backend: full precision (local). Scientist: headless Claude Code (claude-opus-5-5), 57 turns, MCP tools only.

| Published mechanism | Verdict | Evidence in the blind run | Comment |
|---|---|---|---|
| Inputs are embedded as sines and cosines at a sparse set of key frequencies | **found** | E0003 | Top frequencies 28, 17, 2, 14 (plus 27); 9 frequencies hold 92% of embedding power. |
| Attention moves a and b to the '=' position | **found** | E0008, E0009, E0010 | All 4 heads attend about 50/50 to a and b; jointly ablating them removes the behaviour. |
| The MLP combines the two operands into cos/sin of w(a+b) using products (trig identities) | **partial** | E0007, E0011, E0015 | Found: MLP is necessary, neurons are tuned to single key frequencies, MLP output is periodic in the key frequencies. Not shown: the product terms themselves. The run's own test of this step (a probe) failed to execute and it said so. |
| The unembedding reads the answer out as sum over key frequencies of cos(w(a+b-c)) | **partial** | E0004, E0016 | Found: W_U uses the same frequencies, and removing that 12-dim subspace at the output removes the behaviour. The cos(w(a+b-c)) form appears in the pseudocode but was not measured directly on the logits. |
| The key frequencies are causally necessary and sufficient | **found** | E0012, V0001 | 18 of 128 embedding dims: removed -> 0.9% accuracy; kept alone -> 100%; random 18 dims removed -> 100%. |

Counted from the run's records:

- Key frequencies the run reported for the embedding: [2, 4, 14, 17, 27, 28, 34, 43, 56].
- Top 5 frequencies recomputed independently from the weights: [2, 14, 17, 27, 28] (0.823 of embedding power). All reported: True.
- Held-out causal check on the 18 of 128 dim key-frequency subspace of the embedding: accuracy 1.000 with only the subspace, 0.009 with it removed, 1.000 with a random subspace of the same size removed.

Final verification V0001: **verified** on 1000 held-out inputs. Report: [`modadd/report/report.md`](modadd/report/report.md). Notebook: [`modadd/notebook.jsonl`](modadd/notebook.jsonl).

## Induction (2-layer attention-only model)

Ground truth: Elhage, Nanda, Olsson et al. (2021), A Mathematical Framework for Transformer Circuits. The paper describes the mechanism; it does not give head indices for this particular checkpoint (TransformerLens 'attn-only-2l'), so the rubric is at mechanism level.

Backend: full precision (local). Scientist: headless Claude Code (claude-opus-5-5), 72 turns, MCP tools only.

| Published mechanism | Verdict | Evidence in the blind run | Comment |
|---|---|---|---|
| A layer-0 previous-token head | **found** | E0005, E0006 | Head 0.3: 0.53 of attention on the previous position; ablating it costs 8.8 nats. |
| A layer-1 induction head that attends to the token after the earlier occurrence of the current token | **found** | E0003, E0007, E0013 | Head 1.6: 0.64 attention to that token; shown to be content-based by changing the repeat length. |
| K-composition: the previous-token head feeds the induction head's keys | **found** | E0010, E0011 | Path 0.3 -> 1.6 key restores 0.77; the value path restores about 0. |
| The induction head's OV circuit copies the attended token to the output | **found** | E0012 | 1.6 writes 81% of the logit difference directly. |
| The written-out algorithm predicts the model on held-out inputs | **partial** | V0001, V0002 | Causal checks pass, but agreement is 0.605 because the model is only 60% accurate on this task. V0002 used only 10 held-out inputs. |

Counted from the run's records:

- V0001: not-verified (agreement with model 0.605, model accuracy 0.605).
- V0002: not-verified (agreement with model 0.5, model accuracy 0.5).
- Circuit ['0.0', '0.3', '1.6', '1.7']: alone keeps 1.05 of the metric, ablated leaves 0.12; random same-size ablations leave 1.05, 1.07, 0.94.

Final verification V0002: **not-verified** on 10 held-out inputs. Report: [`induction/report/report.md`](induction/report/report.md). Notebook: [`induction/notebook.jsonl`](induction/notebook.jsonl).

## Indirect object identification (GPT-2 small)

Ground truth: Wang, Variengien, Conmy, Shlegeris, Steinhardt (2022), Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. Head lists below are the 26 heads of the published circuit.

Backend: full precision (local). Scientist: headless Claude Code (claude-opus-5-5), 128 turns, MCP tools only.

| Published mechanism | Verdict | Evidence in the blind run | Comment |
|---|---|---|---|
| Name mover heads attend from the last token to the indirect object and copy it | **found** | E0008, E0010, E0020 | 9.9, 9.6, 10.0: all three published heads. |
| Negative name mover heads write against the answer | **found** | E0008, E0010, E0014 | 10.7 and 11.10: both published heads. |
| S-inhibition heads read the repeated subject and steer the name movers' queries | **found** | E0008, E0011, E0023 | 7.3, 7.9, 8.6, 8.10 via path patching into name-mover queries. |
| Induction and duplicate-token heads detect the repeated name | **found** | E0009, E0012, E0024 | Induction 4 of 4; duplicate token 2 of 3 (0.10 missed). |
| Previous-token heads feed the induction heads | **partial** | V0002 | 4.11 is in the final circuit as a supporting head; 2.2 is missing and the previous-token role is not described. |
| Backup name movers take over when name movers are ablated | **partial** | E0013, E0015, E0016, E0018, E0019 | The redundancy was found and measured; 4 of 8 published backup heads are in the circuit. |

Counted from the run's records:

- Recovered circuit: 23 heads plus mlp0, mlp1, mlp2, mlp3, mlp4, mlp5, mlp6. Published circuit: 26 heads. Overlap: 20 found, 6 missed (recall 0.769, precision 0.87).

| Class (Wang et al. 2022) | Published | Found | Missed |
|---|---|---|---|
| Name mover | 9.9, 9.6, 10.0 | 9.9, 9.6, 10.0 | - |
| Negative name mover | 10.7, 11.10 | 10.7, 11.10 | - |
| S-inhibition | 7.3, 7.9, 8.6, 8.10 | 7.3, 7.9, 8.6, 8.10 | - |
| Induction | 5.5, 5.8, 5.9, 6.9 | 5.5, 5.8, 5.9, 6.9 | - |
| Duplicate token | 0.1, 0.10, 3.0 | 0.1, 3.0 | 0.10 |
| Previous token | 2.2, 4.11 | 4.11 | 2.2 |
| Backup name mover | 10.10, 10.6, 10.2, 10.1, 11.2, 9.7, 9.0, 11.9 | 10.10, 10.6, 10.1, 9.7 | 10.2, 11.2, 9.0, 11.9 |

- Heads in the recovered circuit that are not in the published one: 1.11, 4.3, 6.0.
- Held-out causal check: circuit alone keeps 0.99 of the logit difference (top-1 accuracy 0.38); circuit ablated leaves -0.13; random same-size ablations leave 1.05, 1.04, 1.06.

Final verification V0002: **verified** on 200 held-out inputs. Report: [`ioi/report/report.md`](ioi/report/report.md). Notebook: [`ioi/notebook.jsonl`](ioi/notebook.jsonl).

## What this does not show

- The scientist may know these circuits from training. The protocol forbids using recall and the report tool enforces evidence ids, but it cannot prove what guided the choice of experiments.
- One run per benchmark, no repeats, so there is no variance estimate.
- The rubric verdicts (found / partial / missed) were judged afterwards by the developer of this repo (also Claude, not blind), by reading the blind reports. The evidence ids are listed so the judgement can be checked.
- The lab ships a Fourier analysis primitive ('periodicity') and task generators for these three behaviours. The tools do not name mechanisms, but their existence makes some hypotheses cheaper to test than others.
- The agreement check is weak for modular addition: the model is 100% accurate, so any correct description of the task agrees with it. The causal subspace check carries the weight there.
- For induction the model itself is only about 60% accurate on random-token repeats, so an idealised algorithm cannot reach the 90% agreement threshold. That benchmark is reported as partial, as the run itself reported it.
- For IOI the verified circuit keeps 99% of the logit difference on its own, but top-1 accuracy falls to 0.375 (from 0.935). The pass threshold is on the logit difference, so this passed; it is a weaker result than the status alone suggests.
- The blind runs exposed gaps in the lab (logged in the notebooks): direct logit attribution reported heads only, a keep-only subspace ablation was not picked up by the ablation primitive, one linear probe call failed, and a negative metric (log-probability) made the 'fraction of clean' ratio misleading in one verification. The committed results are the original runs, made before any of these were fixed.
