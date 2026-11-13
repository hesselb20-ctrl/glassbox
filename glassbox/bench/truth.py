"""Ground-truth functions of the bench task families, in plain Python (no dependencies).

Each takes the task parameters and the input tokens as strings (without the BOS token) and returns the
output symbol as a string. The same functions are used to check the compiled models, to check the
trained models, and to grade a scientist's recovered algorithm. They live in the repo; the *parameters*
of a generated task live only in its sealed answer key.
"""
from __future__ import annotations


def hist(p: dict, toks: list[str]) -> str:
    """How many tokens of the sequence fall in the same class as the last token."""
    c = p["classes"]
    return str(sum(c[t] == c[toks[-1]] for t in toks))


def sort_by_key(p: dict, toks: list[str]) -> str:
    """The token that comes last when the sequence is sorted by a hidden key (ties: the later one)."""
    key = p["key"]
    best = max(range(len(toks)), key=lambda i: (key[toks[i]], i))
    return toks[best]


def count_compare(p: dict, toks: list[str]) -> str:
    """Count two designated tokens and report which one is more frequent."""
    a, b = toks.count(p["a"]), toks.count(p["b"])
    return p["out"]["gt"] if a > b else p["out"]["eq"] if a == b else p["out"]["lt"]


def shift_copy(p: dict, toks: list[str]) -> str:
    """Take the token k positions before the last one and pass it through a hidden substitution."""
    return p["f"][toks[len(toks) - 1 - p["k"]]]


def lookup2(p: dict, toks: list[str]) -> str:
    """Two-step lookup: the last token points at a position; the token found there, together with the last
    token, selects the answer from a hidden table."""
    fetched = toks[p["ptr"][toks[-1]]]            # positions count from 0 at the first token after BOS
    return p["table"][f"{fetched}|{toks[-1]}"]


def pair_table(p: dict, toks: list[str]) -> str:
    """Trained track: a hidden table indexed by the first and the last token; the tokens in between do not matter."""
    return p["table"][f"{toks[0]}|{toks[-1]}"]


def max_rank(p: dict, toks: list[str]) -> str:
    """Trained track: the token with the highest hidden rank."""
    return max(toks, key=lambda t: p["rank"][t])


FAMILIES = {"hist": hist, "sort-by-key": sort_by_key, "count-compare": count_compare, "shift-copy": shift_copy,
            "lookup2": lookup2, "pair-table": pair_table, "max-rank": max_rank}
COMPILED = ("hist", "sort-by-key", "count-compare", "shift-copy", "lookup2")
TRAINED = ("pair-table", "max-rank")


def truth(family: str, params: dict, toks: list[str]) -> str:
    return FAMILIES[family](params, toks)
