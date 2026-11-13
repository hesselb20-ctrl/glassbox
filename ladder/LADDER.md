# Scale ladder: indirect object identification

In GPT-2 small all four predictions hold at full precision (gpt2-local, gpt2-ndif). In the official Qwen3.8-27B screened in 4-bit on two 12 GB GPUs, 4 of 4 hold. At full precision on NDIF, 4 of 4 hold, which confirms the screen. So the same three-stage structure (the repeated name is tracked at S2, handed to the last token late in the network, and written out by a few backed-up attention heads) appears at about 220 times the size, at a similar relative depth. These are four coarse predictions, not the full 26-head circuit. An earlier NDIF run is kept in qwen27b-ndif-first-run: with only 6 prompts (a memory bug in glassbox, since fixed, limited the batch size) its behaviour check missed the accuracy threshold (0.50); the other three predictions held there too.

The four predictions come from the circuit the blind benchmark recovered in GPT-2 small. Each rung runs the same script (`ladder/run_ladder.py`). Quantized rungs are screening only.

| Rung | Model | Backend / precision | P1 behaviour | P2 hand-off | P3 name movers | P4 backup | Status |
|---|---|---|---|---|---|---|---|
| gpt2-local | gpt2 (124M) | full precision (local) | supported | supported | supported | supported | **confirmed** |
| gpt2-ndif | openai-community/gpt2 (124M) | full precision (NDIF remote) | supported | supported | supported | supported | **confirmed** |
| qwen27b-gguf | Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf (27B) | screening (gguf quantized, abliterated variant) | supported | not testable on this backend | not testable on this backend | not testable on this backend | **partly tested, supported so far (screening)** |
| qwen27b-4bit | Qwen/Qwen3.8-27B (27B) | screening (4-bit) | supported | supported | supported | supported | **screening-pass** |
| qwen27b-ndif | Qwen/Qwen3.8-27B (27B) | full precision (NDIF remote) | supported | supported | supported | supported | **confirmed** |

## Numbers

- **gpt2-local** (full precision (local)): accuracy 0.91, logit diff 3.8; S2 to END hand-off at layer 9/11; top heads 9.9, 9.6, 7.9 restore 1.19; ablating them leaves 0.85.
- **gpt2-ndif** (full precision (NDIF remote)): accuracy 0.81, logit diff 3.7; S2 to END hand-off at layer 9/11; top heads 9.9, 9.6, 7.3 restore 1.32; ablating them leaves 0.92.
- **qwen27b-gguf** (screening (gguf quantized, abliterated variant)): accuracy 0.81, logit diff 6.2.
- **qwen27b-4bit** (screening (4-bit)): accuracy 0.88, logit diff 6.3; S2 to END hand-off at layer 55/63; top heads 59.14, 59.8, 59.9 restore 0.36; ablating them leaves 1.00.
- **qwen27b-ndif** (full precision (NDIF remote)): accuracy 0.81, logit diff 6.0; S2 to END hand-off at layer 55/63; top heads 59.14, 59.8, 59.9 restore 0.37; ablating them leaves 1.00.

Each rung folder holds the full notebook, experiment records and plots.
