# Scale ladder: indirect object identification

In GPT-2 small all four predictions hold at full precision (gpt2-local, gpt2-ndif). In the official Qwen3.8-27B at full precision on NDIF, 3 of 4 hold: the S2-to-END hand-off, a few late heads writing the answer, backup. Not met: the behaviour itself (top-1 accuracy 0.50 on only 6 prompts, below the preset 0.8, although the logit difference is 5.4 in favour of the indirect object). So the same three-stage structure shows up at 220 times the size, at a similar relative depth, but this rung is small (NDIF's replica had little spare memory) and is not a full confirmation. Not done yet: qwen27b-4bit.

The four predictions come from the circuit the blind benchmark recovered in GPT-2 small. Each rung runs the same script (`ladder/run_ladder.py`). Quantized rungs are screening only.

| Rung | Model | Backend / precision | P1 behaviour | P2 hand-off | P3 name movers | P4 backup | Status |
|---|---|---|---|---|---|---|---|
| gpt2-local | gpt2 (124M) | full precision (local) | supported | supported | supported | supported | **confirmed** |
| gpt2-ndif | openai-community/gpt2 (124M) | full precision (NDIF remote) | supported | supported | supported | supported | **confirmed** |
| qwen27b-gguf | Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf (27B) | screening (gguf quantized, abliterated variant) | supported | not testable on this backend | not testable on this backend | not testable on this backend | **partly tested, supported so far (screening)** |
| qwen27b-ndif | Qwen/Qwen3.8-27B (27B) | full precision (NDIF remote) | rejected | supported | supported | supported | **partly supported** |
| qwen27b-4bit | Qwen/Qwen3.8-27B (27B) | local-4bit | - | - | - | - | not done yet |

## Numbers

- **gpt2-local** (full precision (local)): accuracy 0.91, logit diff 3.8; S2 to END hand-off at layer 9/11; top heads 9.9, 9.6, 7.9 restore 1.19; ablating them leaves 0.85.
- **gpt2-ndif** (full precision (NDIF remote)): accuracy 0.81, logit diff 3.7; S2 to END hand-off at layer 9/11; top heads 9.9, 9.6, 7.3 restore 1.32; ablating them leaves 0.92.
- **qwen27b-gguf** (screening (gguf quantized, abliterated variant)): accuracy 0.81, logit diff 6.2.
- **qwen27b-ndif** (full precision (NDIF remote)): accuracy 0.50, logit diff 5.4; S2 to END hand-off at layer 55/63; top heads 59.8, 59.6, 51.23 restore 0.67; ablating them leaves 0.85.

Each rung folder holds the full notebook, experiment records and plots.
