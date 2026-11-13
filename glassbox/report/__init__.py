"""Write the investigation up: report.md plus a self-contained report.html.

The report refuses to be written if any claim does not cite an experiment that actually ran.
Every number in it is shown next to the backend and precision that produced it.
"""
from __future__ import annotations

import base64
import html
import json
import time
from pathlib import Path
from typing import Any

from .. import style as S


class UnbackedClaim(ValueError):
    pass


def _records(inv: Path) -> dict[str, dict]:
    out = {}
    for p in sorted((inv / "experiments").glob("[EV]*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["id"]] = r
    return out


def circuit_svg(circuit: dict, width: int = 920) -> str:
    """Layered left-to-right diagram. nodes: {id, label, layer, kind}; edges: {src, dst, label}."""
    nodes = circuit.get("nodes", [])
    if not nodes:
        return ""
    layers = sorted({n.get("layer", 0) for n in nodes})
    cols: dict[Any, list] = {l: [n for n in nodes if n.get("layer", 0) == l] for l in layers}
    rows = max(len(v) for v in cols.values())
    bw, bh, pad = 150, 44, 28
    height = pad * 2 + rows * (bh + 22)
    xs = {l: pad + (i * (width - 2 * pad - bw) / max(1, len(layers) - 1) if len(layers) > 1 else (width - bw) / 2 - pad)
          for i, l in enumerate(layers)}
    pos = {}
    for l, col in cols.items():
        gap = (height - 2 * pad - len(col) * bh) / (len(col) + 1)
        for i, n in enumerate(col):
            pos[n["id"]] = (xs[l], pad + gap * (i + 1) + bh * i)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
           f'font-family="Consolas, DejaVu Sans Mono, monospace">',
           f'<defs><filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/>'
           f'<feMergeNode in="SourceGraphic"/></feMerge></filter>'
           f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{S.ACCENT}"/></marker></defs>',
           f'<rect width="{width}" height="{height}" fill="{S.NAVY}" rx="10"/>']
    for e in circuit.get("edges", []):
        if e["src"] not in pos or e["dst"] not in pos:
            continue
        (x1, y1), (x2, y2) = pos[e["src"]], pos[e["dst"]]
        x1, y1, y2 = x1 + bw, y1 + bh / 2, y2 + bh / 2
        mx = (x1 + x2) / 2
        out.append(f'<path d="M{x1:.0f},{y1:.0f} C{mx:.0f},{y1:.0f} {mx:.0f},{y2:.0f} {x2 - 3:.0f},{y2:.0f}" fill="none" '
                   f'stroke="{S.ACCENT}" stroke-width="2" filter="url(#glow)" marker-end="url(#arr)" opacity="0.9"/>')
        if e.get("label"):
            out.append(f'<text x="{mx:.0f}" y="{(y1 + y2) / 2 - 6:.0f}" fill="{S.MUTED}" font-size="11" '
                       f'text-anchor="middle">{html.escape(str(e["label"]))}</text>')
    for n in nodes:
        x, y = pos[n["id"]]
        io = n.get("kind") in ("input", "output")
        out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw}" height="{bh}" rx="8" fill="{S.PANEL}" '
                   f'stroke="{S.MUTED if io else S.ACCENT}" stroke-width="1.5"/>')
        out.append(f'<text x="{x + bw / 2:.0f}" y="{y + bh / 2 + 4:.0f}" fill="{S.TEXT}" font-size="12" '
                   f'text-anchor="middle">{html.escape(str(n.get("label", n["id"])))[:22]}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _verification_rows(v: dict) -> list[tuple[str, str]]:
    rows = [("status", v["status"]), ("held-out inputs", str(v["n_heldout"])), ("model accuracy", _fmt(v["model_accuracy"]))]
    c = v.get("checks", {})
    if "agreement" in c:
        rows += [("algorithm agrees with model", _fmt(c["agreement"]["agreement_with_model"])),
                 ("algorithm accuracy vs ground truth", _fmt(c["agreement"]["algorithm_accuracy_vs_ground_truth"]))]
    if "causal_components" in c:
        k = c["causal_components"]
        rows += [(f"circuit size", f"{k['circuit_size']} of {k['total_components']} components"),
                 ("metric kept with ONLY the circuit", _fmt(k["everything_else_ablated"]["fraction_of_clean"])),
                 ("metric kept with the circuit ABLATED", _fmt(k["circuit_ablated"]["fraction_of_clean"])),
                 ("random same-size ablation (control)", ", ".join(_fmt(x) for x in k["random_same_size_ablated_fraction"]))]
    if "causal_subspace" in c:
        k = c["causal_subspace"]
        rows += [("subspace size", f"{sum(k['dims'])} of {k['of_total_dims'][0]} dims"),
                 ("accuracy with ONLY the subspace", _fmt(k["only_subspace_kept"]["accuracy"])),
                 ("accuracy with the subspace REMOVED", _fmt(k["subspace_removed"]["accuracy"])),
                 ("accuracy with a random subspace removed (control)", _fmt(k["random_subspace_removed"]["accuracy"]))]
    return rows


def write_report(inv: Path, report: dict, notebook, state: dict) -> dict:
    inv = Path(inv)
    recs = _records(inv)
    claims = report.get("claims", [])
    bad = []
    for i, c in enumerate(claims):
        ev = c.get("evidence") or []
        missing = [e for e in ev if e not in recs]
        if not ev or missing:
            bad.append(f"claim {i + 1} ({c.get('text', '')[:60]!r}): " + ("no evidence cited" if not ev else f"unknown ids {missing}"))
    if bad:
        raise UnbackedClaim("Every claim must cite experiments that ran in this investigation. Problems: " + "; ".join(bad))
    # exploratory runs (no pre-registration) may illustrate a claim but can never be its only support
    weak = [f"claim {i + 1} ({c.get('text', '')[:60]!r})" for i, c in enumerate(claims)
            if all(recs[e].get("exploratory") for e in c["evidence"])]
    if weak:
        raise UnbackedClaim("These claims rest only on exploratory runs. Each claim needs at least one pre-registered experiment "
                            "or a verification among its evidence: " + "; ".join(weak))
    pr = notebook.prediction_record()
    exploratory = [r for r in recs.values() if r.get("exploratory") and r.get("primitive") != "dataset"]
    legacy = [r for r in recs.values() if r["id"].startswith("E") and "exploratory" not in r]

    cited = sorted({e for c in claims for e in c["evidence"]})
    key_plots = [p for p in report.get("key_plots", []) if p in recs and recs[p].get("plot")]
    verifs = [r for r in recs.values() if r["id"].startswith("V")]
    tags = sorted({r["provenance"]["tag"] for r in recs.values()})
    title = report.get("title", "Glassbox investigation")
    out_dir = inv / "report"
    out_dir.mkdir(exist_ok=True)
    final = [v for v in verifs if v["id"] in report.get("verification", [v["id"] for v in verifs])]
    statuses = [v["status"] for v in final]
    overall = ("verified" if statuses and all(s == "verified" for s in statuses) else
               "screening" if statuses and all(s in ("verified", "screening-pass") for s in statuses) else
               "partial / not verified" if statuses else "unverified (no verification run)")
    if report.get("label"):
        overall = f"{overall} · {report['label']}"

    # ---------------------------------------------------------------- markdown
    md = [f"# {title}", "",
          f"**Behavior:** {report.get('behavior', '')}  ",
          f"**Model:** `{state.get('model')}` on `{state.get('backend')}`  ",
          f"**Status:** {overall}  ",
          f"**Experiments run:** {sum(1 for r in recs if r.startswith('E'))} · **Backends used:** {', '.join(tags)}  ",
          f"**Written:** {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if report.get("summary"):
        md += ["## Summary", "", report["summary"], ""]
    if report.get("pseudocode"):
        md += ["## The recovered algorithm", "", "```python", report["pseudocode"].rstrip(), "```", ""]
    if report.get("circuit", {}).get("nodes"):
        (out_dir / "circuit.svg").write_text(circuit_svg(report["circuit"]), encoding="utf-8")
        md += ["## Circuit", "", "![circuit](circuit.svg)", ""]
    if claims:
        md += ["## Claims and evidence", "", "| # | Claim | Evidence | Produced by |", "|---|---|---|---|"]
        for i, c in enumerate(claims, 1):
            by = sorted({recs[e]["provenance"]["tag"] for e in c["evidence"]})
            md.append(f"| {i} | {c['text']} | {', '.join(c['evidence'])} | {'; '.join(by)} |")
        md.append("")
    if final:
        md += ["## Verification", ""]
        for v in final:
            md += [f"### {v['id']}: {v['name']}  ({v['provenance']['tag']})", "", "| check | value |", "|---|---|"]
            md += [f"| {a} | {b} |" for a, b in _verification_rows(v)]
            md += ["", v["note"], "", f"![{v['id']}](../experiments/{Path(v['plot']).name})", ""]
    if key_plots:
        md += ["## Key plots", ""]
        for p in key_plots:
            r = recs[p]
            md += [f"**{p}: {r['primitive']}** ({r['provenance']['tag']})", "",
                   f"![{p}](../experiments/{Path(r['plot']).name})", ""]
    for head, key in (("Limitations", "limitations"), ("Open questions", "open_questions")):
        if report.get(key):
            md += [f"## {head}", ""] + [f"- {x}" for x in report[key]] + [""]
    md += ["## Prediction record", ""]
    if pr["preregistered"]:
        hr = "n/a" if pr["hit_rate"] is None else f"{pr['hit_rate'] * 100:.0f}%"
        md += [f"{pr['preregistered']} experiments were pre-registered, {pr['run']} were run: **{pr['confirmed']} confirmed, "
               f"{pr['refuted']} refuted, {pr['inconclusive']} inconclusive**. Hit rate (confirmed / decided): **{hr}**.", ""]
        if pr["refuted_predictions"]:
            md += ["Refuted predictions (shown, not hidden):", "", "| prereg | experiment | hypothesis | predicted | observed |", "|---|---|---|---|---|"]
            md += [f"| {x['prereg']} | {x['experiment']} | {x['hypothesis'][:160]} | `{x['prediction']['metric']} {x['prediction']['op']} "
                   f"{x['prediction']['threshold']}` | {_fmt(x['observed'])} |" for x in pr["refuted_predictions"]]
            md.append("")
        else:
            md += ["No prediction was refuted.", ""]
    else:
        md += ["No experiments were pre-registered in this investigation" + (" (it predates pre-registration)." if legacy else "."), ""]
    if exploratory:
        md += [f"Exploratory runs (no pre-registration; they cannot be the only support for a claim): "
               f"{', '.join(r['id'] for r in exploratory)}.", ""]
    md += ["## Experiments", "", "| id | primitive | backend / precision | seconds | pre-registration |", "|---|---|---|---|---|"]

    def _how(r):
        if r["id"].startswith("V"):
            return "verification"
        if r.get("prereg_id"):
            return f"{r['prereg_id']}: {(r.get('score') or {}).get('verdict')}"
        return "exploratory" if r.get("exploratory") else "none (older run)"

    md += [f"| {r['id']} | {r.get('primitive', 'verify')} | {r['provenance']['tag']} | {r.get('seconds', '')} | {_how(r)} |" for r in recs.values()]
    md += ["", "## Notebook trail", ""]
    for e in notebook.view():
        st = f" **[{e['status']}]**" if e.get("status") else ""
        refs = f" ({', '.join(e['refs'])})" if e.get("refs") else ""
        tgt = f" -> {e['target']}" if e.get("target") else ""
        md.append(f"- `{e['id']}` *{e['kind']}*{st}{tgt}: {e['text']}{refs}")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- html
    def img(path: str) -> str:
        b = base64.b64encode((inv / path).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b}" alt=""/>'

    esc = html.escape
    h = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>
body{{background:{S.NAVY};color:{S.TEXT};font:15px/1.6 Inter,Segoe UI,system-ui,sans-serif;margin:0}}
main{{max-width:980px;margin:0 auto;padding:40px 20px 80px}}
h1{{font-size:30px;margin:0 0 6px}} h2{{font-size:19px;margin:44px 0 12px;color:{S.ACCENT};font-weight:600}}
h3{{font-size:15px;margin:22px 0 8px}} code,pre,.mono,td.id{{font-family:Consolas,'DejaVu Sans Mono',monospace}}
pre{{background:{S.PANEL};border:1px solid {S.GRID};border-radius:8px;padding:16px;overflow-x:auto;font-size:13.5px;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid {S.GRID};vertical-align:top}}
th{{color:{S.MUTED};font-weight:500}} img{{max-width:100%;border-radius:8px;border:1px solid {S.GRID}}}
.meta{{color:{S.MUTED};font-size:13.5px}} .tag{{display:inline-block;border:1px solid {S.GRID};border-radius:99px;padding:1px 9px;font-size:12px;color:{S.MUTED};margin-right:4px;white-space:nowrap}}
.status{{color:{S.ACCENT};border-color:{S.ACCENT}}} .card{{background:{S.PANEL};border:1px solid {S.GRID};border-radius:10px;padding:16px;margin:12px 0}}
.rejected{{color:{S.WARN}}} .supported{{color:{S.ACCENT}}} .trail div{{padding:5px 0;border-bottom:1px solid {S.GRID};font-size:13.5px}}
.wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>{esc(title)}</h1>
<p class="meta">{esc(report.get('behavior', ''))}</p>
<p><span class="tag status">{esc(overall)}</span><span class="tag">model: {esc(str(state.get('model')))}</span>
<span class="tag">{sum(1 for r in recs if r.startswith('E'))} experiments</span>{''.join(f'<span class="tag">{esc(t)}</span>' for t in tags)}</p>"""]
    if report.get("summary"):
        h.append(f"<h2>Summary</h2><p>{esc(report['summary'])}</p>")
    if report.get("pseudocode"):
        h.append(f"<h2>The recovered algorithm</h2><pre>{esc(report['pseudocode'].rstrip())}</pre>")
    if report.get("circuit", {}).get("nodes"):
        h.append(f"<h2>Circuit</h2>{circuit_svg(report['circuit'])}")
    if claims:
        h.append('<h2>Claims and evidence</h2><div class="wrap"><table><tr><th>#</th><th>Claim</th><th>Evidence</th><th>Produced by</th></tr>')
        for i, c in enumerate(claims, 1):
            by = sorted({recs[e]["provenance"]["tag"] for e in c["evidence"]})
            h.append(f"<tr><td>{i}</td><td>{esc(c['text'])}</td><td class='id'>{', '.join(c['evidence'])}</td>"
                     f"<td>{''.join(f'<span class=tag>{esc(b)}</span>' for b in by)}</td></tr>")
        h.append("</table></div>")
    if final:
        h.append("<h2>Verification</h2>")
        for v in final:
            h.append(f"<div class='card'><h3>{v['id']}: {esc(v['name'])} <span class='tag'>{esc(v['provenance']['tag'])}</span></h3><table>")
            h += [f"<tr><td>{esc(a)}</td><td class='mono'>{esc(b)}</td></tr>" for a, b in _verification_rows(v)]
            h.append(f"</table><p class='meta'>{esc(v['note'])}</p>{img(v['plot'])}</div>")
    if key_plots:
        h.append("<h2>Key plots</h2>")
        for p in key_plots:
            r = recs[p]
            h.append(f"<div class='card'><h3>{p}: {esc(r['primitive'])} <span class='tag'>{esc(r['provenance']['tag'])}</span></h3>{img(r['plot'])}</div>")
    for head, key in (("Limitations", "limitations"), ("Open questions", "open_questions")):
        if report.get(key):
            h.append(f"<h2>{head}</h2><ul>" + "".join(f"<li>{esc(x)}</li>" for x in report[key]) + "</ul>")
    h.append("<h2>Prediction record</h2>")
    if pr["preregistered"]:
        hr = "n/a" if pr["hit_rate"] is None else f"{pr['hit_rate'] * 100:.0f}%"
        h.append(f"<p>{pr['preregistered']} pre-registered, {pr['run']} run: <span class='supported'>{pr['confirmed']} confirmed</span>, "
                 f"<span class='rejected'>{pr['refuted']} refuted</span>, {pr['inconclusive']} inconclusive. Hit rate <b>{hr}</b>.</p>")
        if pr["refuted_predictions"]:
            h.append('<div class="wrap"><table><tr><th>prereg</th><th>experiment</th><th>hypothesis</th><th>predicted</th><th>observed</th></tr>')
            h += [f"<tr><td class='id'>{x['prereg']}</td><td class='id'>{x['experiment']}</td><td>{esc(x['hypothesis'][:200])}</td>"
                  f"<td class='mono'>{esc(str(x['prediction']['metric']))} {esc(x['prediction']['op'])} {esc(str(x['prediction']['threshold']))}</td>"
                  f"<td class='mono rejected'>{esc(_fmt(x['observed']))}</td></tr>" for x in pr["refuted_predictions"]]
            h.append("</table></div>")
    else:
        h.append("<p class='meta'>No experiments were pre-registered in this investigation.</p>")
    if exploratory:
        h.append(f"<p class='meta'>Exploratory runs (cannot be the only support for a claim): {', '.join(r['id'] for r in exploratory)}.</p>")
    h.append('<h2>Experiments</h2><div class="wrap"><table><tr><th>id</th><th>primitive</th><th>backend / precision</th><th>seconds</th><th>pre-registration</th></tr>')
    h += [f"<tr><td class='id'>{r['id']}</td><td>{esc(r.get('primitive', 'verify'))}</td><td>{esc(r['provenance']['tag'])}</td>"
          f"<td>{r.get('seconds', '')}</td><td>{esc(_how(r))}</td></tr>" for r in recs.values()]
    h.append('</table></div><h2>Notebook trail</h2><div class="trail">')
    for e in notebook.view():
        st = f" <span class='{e['status']}'>[{e['status']}]</span>" if e.get("status") else ""
        refs = f" <span class='meta mono'>({', '.join(e['refs'])})</span>" if e.get("refs") else ""
        tgt = f" → {e['target']}" if e.get("target") else ""
        h.append(f"<div><span class='mono meta'>{e['id']} {e['kind']}{tgt}</span>{st} {esc(e['text'])}{refs}</div>")
    h.append("</div></main></body></html>")
    (out_dir / "report.html").write_text("\n".join(h), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({**report, "overall_status": overall, "cited": cited, "prediction_record": pr,
                                                      "exploratory_runs": [r["id"] for r in exploratory]}, indent=1), encoding="utf-8")
    return {"markdown": "report/report.md", "html": "report/report.html", "status": overall, "cited": cited,
            "prediction_record": {k: pr[k] for k in ("preregistered", "run", "confirmed", "refuted", "inconclusive", "hit_rate")}}
