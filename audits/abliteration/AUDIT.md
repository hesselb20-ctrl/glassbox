# Audit: abliterated vs official Qwen3.8-27B

**Label: screening (quantized GGUF).** Both files are about 4.8 bits per weight. This is a measurement of where two models differ. No completions were generated or stored; refusal is scored from the probability of the first answer token.

Files: `Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf` (community, abliterated) and `Qwen3.8-27B-UD-Q4_K_M.gguf` (official weights, quantized by unsloth). Noise floor: `Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf`, the same abliterated weights in a different quantization.

## Bottom line

The official model starts a refusal on 100% of the harmful instructions, the abliterated one on 2%. 5 of the 8 pre-registered predictions were refuted: the difference is not the textbook picture of one direction removed from every writing matrix. An exploratory follow-up (below, not pre-registered) places the edit in 52 matrices, the MLP output and linear-attention output of layers 22 to 51, all changed along one direction whose cosine with the official model's refusal direction is 0.87 at layer 34. Confidence: high for the location, moderate for the identity of the direction. Label: screening (quantized GGUF).

## Result of the pre-registered analysis

- **Behaviour.** Refusal-start rate on 64 harmful instructions: official 1.00, abliterated 0.02 (harmless: 0.05 and 0.02).
- **Which matrices.** 60% of the 130 matrices that write to the residual stream (attention output, MLP output) have a difference whose top component stands above the 95th percentile of the null (0.0052); median top-component energy 0.014 against 0.0010 for matrices that read from it and 0.0011 for pure quantization noise. 0% of the 130 reading matrices exceed the noise floor.
- **Which layers.** Edited layers: 54 of 64 (0 to 56).
- **Rank.** Median ratio of the first to the second singular value of the difference: 2.7 in writing matrices, 1.1 in reading matrices. The raw difference is not rank 1 because quantization noise is larger in norm than the edit: median relative difference 0.080 (noise floor between two quantizations of the same weights: 0.154).
- **One shared direction.** Median absolute cosine between the top directions of different writing matrices: 0.03 (reading matrices: 0.038, 95th percentile 0.386).
- **Is it the refusal direction?** Cosine between the weight-derived direction and the direction that separates harmful from harmless prompts in the official model's activations (layer 62): **0.08** (maximum over layers 0.51). Nulls, 95th percentile: random directions 0.026 (n=2000), unrelated difference-of-means directions 0.243 (n=200), top directions of unedited matrices 0.028 (n=130).
- **Where the signal vanishes.** Along the official model's direction, the gap between harmful and harmless prompts in the abliterated model is below a quarter of the official model's in 30 of 64 layers (median ratio 0.48; first such layer: 33).

## Pre-registered predictions

3 confirmed, 5 refuted. Predictions were written to the hash-chained notebook before any measurement.

| | Hypothesis | Prediction | Observed | Verdict |
|---|---|---|---|---|
| N0003 | The official model refuses the harmful instructions and not the harmless ones. | `refusal_rate_harmful >= 0.8` | 1.000 | **confirmed** |
| N0004 | The abliterated model no longer refuses the same harmful instructions. | `refusal_rate_harmful <= 0.2` | 0.016 | **confirmed** |
| N0005 | The weight differences are structured only in matrices that write to the residual stream. | `writers_above_null_share >= 0.8` | 0.600 | **refuted** |
| N0006 | Matrices that only read from the residual stream were not edited. | `readers_above_noise_share <= 0.1` | 0.000 | **confirmed** |
| N0007 | In the edited matrices the difference is close to rank 1 once quantization noise is accounted for. | `writers_median_sigma1_over_sigma2 >= 3.0` | 2.706 | **refuted** |
| N0008 | All edited matrices were changed along one shared direction of the residual stream. | `writers_median_pairwise_cosine >= 0.8` | 0.027 | **refuted** |
| N0009 | That shared direction is the refusal direction found independently from the official model's activations. | `cosine_weight_vs_activation >= 0.5` | 0.076 | **refuted** |
| N0010 | In the abliterated model the refusal signal along the official model's direction has vanished. | `median_signal_ratio <= 0.25` | 0.713 | **refuted** |

## What this does and does not show

- It locates the edit (which matrices, which layers, one direction) from the two files alone, and checks it against a direction found independently from activations.
- Both files are quantized, by different people with different schemes. The noise floor is measured, not assumed; the edit is detected by its spectrum, not by its size.
- The refusal score is a first-token proxy: a refusal usually starts with one of a few tokens. It is not a judgement of full answers.
- Difference of means at the last prompt token is one way to find a refusal direction; the direction the editor actually removed may have been computed at another layer or position.
- No direction vector, prompt or completion is published, and no weights were modified.

Per-matrix numbers: `weights.json`. Per-layer numbers: `summary.json`. Notebook: `notebook.jsonl`.

## Exploratory follow-up (after the predictions were scored)

Five of the eight predictions failed, so the registered picture (every writing matrix edited along one direction) is wrong for this file. This section looks at what the difference is instead. It was designed after seeing the results above and is **not** a pre-registered test.

- **The pooled direction was a quantization artefact.** Two quantizations of the *same* weights also differ along one preferred direction in writing matrices (noise-floor top-component energy up to 0.079). 76 of the 128 writing matrices carry only that: their energy is 0.9 times their own noise floor (median) and their top direction has cosine 0.97 with the artefact direction taken from the noise floor alone.
- **Where the edit is.** 52 matrices stand clear of their own noise floor (median 11 times, minimum 5.2 times) and share one direction (minimum cosine with it 0.997; its cosine with the artefact direction is 0.02). They are in layers 22 to 51 of 64, 30 layers. By tensor: `attn_output.weight` 0 of 16; `ffn_down.weight` 30 of 64; `ssm_out.weight` 22 of 48.
- **What was left alone.** Layers 0 to 21 and 52 to 63. No reading matrix carries the edit direction (largest cosine 0.04). Inside the edited range, 8 writing matrices are untouched: `blk.23.attn_output.weight`, `blk.27.attn_output.weight`, `blk.31.attn_output.weight`, `blk.35.attn_output.weight`, `blk.39.attn_output.weight`, `blk.43.attn_output.weight`, `blk.47.attn_output.weight`, `blk.51.attn_output.weight`. 4 tensors of an extra block (`blk.64`) are byte-identical in both files.
- **Unexplained.** 30 reading matrices, 26 of them inside the edited range, have a faint structure of their own: a top component of 0.2% to 1.2% of the difference, 3 to 11 times their noise floor, along directions unrelated to the edit direction. The noise floor comes from a different pair of quantization schemes than the comparison itself, so this may be a property of the schemes; it may also be a second, much smaller change.
- **Is it the refusal direction?** Cosine between the edit direction and the official model's harmful-minus-harmless direction: **0.87 at layer 34** (the layer was picked as the best of 64). Nulls with the best layer picked the same way: top directions of the 128 reading matrices, 95th percentile 0.05 (maximum 0.09); 300 random directions, 95th percentile 0.04; the artefact direction, 0.51. Unrelated difference-of-means directions at layer 34: 95th percentile 0.16 (n=500).
- **Activations agree on the place.** Along the official direction, the abliterated model keeps 0.99 of the harmful-versus-harmless gap before layer 22, falls below a quarter at layer 33, reaches its minimum (0.10) at layer 34, and stays at 0.19 (median) after that. The signal is reduced, not gone: along its own best direction the abliterated model still separates the two classes at the last layer (d' 4.2, official 10.6).

### Confidence

- **Location (which matrices and layers): high.** It rests on 52 matrices that agree on one direction (cosine above 0.99)  and on an independent activation measurement that changes at the same depth. It does not depend on the choice of prompts.
- **Identity (it is the refusal direction): moderate.** A cosine of 0.87 against nulls below 0.51 is a clear match, and refusal behaviour is gone. But the layer was chosen after the fact, the direction comes from 64 prompts per class at one token position, and 25% of the edit direction's variance is not explained by it.
- **Everything is a screening result.** Both files are quantized by different people; quantization noise is larger in norm than the edit in every matrix. Edits smaller than the noise floor, if any, cannot be seen. This audit was not repeated on unquantized weights.
- The audit says where two files differ and along what. It does not say how the edit was made, and it cannot rule out further changes that leave no low-rank trace.

