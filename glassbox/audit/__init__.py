"""Model diffing: locate an edit between two versions of a model without being told where it is.

Two independent views of the same pair of GGUF files:

* ``weights``      compares matching weight matrices one at a time (dequantized on the fly) and asks, for each,
                   how big the difference is, how concentrated it is in one direction, and which direction of the
                   residual stream that is;
* ``activations``  runs both models on a contrast set of prompts and compares the directions that separate the
                   two prompt classes, layer by layer.

This is an audit tool. It measures where two models differ and whether a known kind of edit is present.
It stores statistics (norms, cosines, scores per prompt), never model completions, and it does not write
modified weights.
"""
