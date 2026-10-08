"""Self-contained HTML evaluation report (no JS/CSS dependencies, light + dark mode)."""

from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
:root{color-scheme:light;
  --page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
  --series-1:#2a78d6;--series-2:#eb6834;--good:#006300;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
  --page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --series-1:#3987e5;--series-2:#d95926;--good:#0ca30c;--critical:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;
  --page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --series-1:#3987e5;--series-2:#d95926;--good:#0ca30c;--critical:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px;font-weight:650}
h2{font-size:17px;margin:0 0 4px;font-weight:620}
.sub{color:var(--text-secondary);margin:0}
.lede{font-size:17px;margin:20px 0 24px;max-width:72ch}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:20px;margin:0 0 20px}
.card .sub{margin-bottom:16px;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:0 0 20px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px}
.tile .label{color:var(--text-secondary);font-size:13px}
.tile .value{font-size:30px;font-weight:620;margin:2px 0}
.tile .note{color:var(--muted);font-size:12.5px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:20px}
.grid2>.card{margin:0;min-width:0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:560}
.scroll{overflow-x:auto}
.cm td.cell{text-align:center;min-width:64px;border:2px solid var(--surface-1)}
.cm th.rot{text-align:center}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--text-secondary);margin:0 0 8px}
.key{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
svg{display:block;max-width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font-size:12px}
svg .lbl{fill:var(--text-secondary)}
svg .val{fill:var(--text-primary);font-size:11.5px;font-variant-numeric:tabular-nums}
[data-tip]{cursor:default}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);
  border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12.5px;line-height:1.45;
  box-shadow:0 4px 16px rgba(0,0,0,.12);z-index:10;white-space:pre;max-width:320px}
details summary{cursor:pointer;color:var(--text-secondary);margin-top:8px}
code{font-size:13px}
.foot{color:var(--muted);font-size:12.5px}
"""

JS = """
const tip=document.getElementById('tip');
function show(e,text){tip.textContent=text;tip.hidden=false;
  const w=tip.offsetWidth,h=tip.offsetHeight;let x=e.clientX+14,y=e.clientY+14;
  if(x+w>innerWidth-8)x=e.clientX-w-14; if(y+h>innerHeight-8)y=e.clientY-h-14;
  tip.style.left=x+'px';tip.style.top=y+'px'}
document.addEventListener('pointermove',e=>{
  const t=e.target.closest('[data-tip]'); if(t){show(e,t.dataset.tip);return}
  if(!e.target.closest('.roc-hit'))tip.hidden=true});
document.querySelectorAll('svg.roc').forEach(svg=>{
  const d=JSON.parse(svg.dataset.roc),hit=svg.querySelector('.roc-hit'),
        cross=svg.querySelector('.cross'),dot=svg.querySelector('.cross-dot'),
        X=+svg.dataset.x0,W=+svg.dataset.w,Y=+svg.dataset.y0,H=+svg.dataset.h;
  hit.addEventListener('pointermove',e=>{
    const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
    const q=p.matrixTransform(svg.getScreenCTM().inverse()),f=Math.min(1,Math.max(0,(q.x-X)/W));
    let i=0;for(let k=0;k<d.fpr.length;k++){if(Math.abs(d.fpr[k]-f)<Math.abs(d.fpr[i]-f))i=k}
    const cx=X+d.fpr[i]*W,cy=Y+H-d.tpr[i]*H;
    cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.style.display='';
    dot.setAttribute('cx',cx);dot.setAttribute('cy',cy);dot.style.display='';
    show(e,'False-alarm rate '+(d.fpr[i]*100).toFixed(1)+'%\\nDetection rate '+(d.tpr[i]*100).toFixed(1)+'%')});
  hit.addEventListener('pointerleave',()=>{cross.style.display='none';dot.style.display='none';tip.hidden=true})});
"""


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _pct(v, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def _tile(label: str, value: str, note: str = "") -> str:
    return (f'<div class="tile"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>'
            f'<div class="note">{_e(note)}</div></div>')


def _bar_path(x0: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x0:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z")


def _hbars(rows: list[dict], *, label_w: int = 150, width: int = 640, bar_h: int = 14,
           row_h: int = 22, fmt=_pct) -> str:
    """rows: {label, value (0..1), color (css var), tip}. One shared 0..1 axis."""
    plot_w = width - label_w - 56
    top, n = 22, len(rows)
    height = top + n * row_h + 6
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" width="{width}">']
    for t in (0, .25, .5, .75, 1):
        x = label_w + t * plot_w
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{height - 4}" '
                   f'stroke="var({"--axis" if t == 0 else "--grid"})" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top - 9}" text-anchor="middle">{int(t * 100)}%</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h + (row_h - bar_h) / 2
        w = max(r["value"], 0) * plot_w
        out.append(f'<g data-tip="{_e(r["tip"])}">'
                   f'<rect x="0" y="{top + i * row_h}" width="{width}" height="{row_h}" fill="transparent"/>'
                   f'<text class="lbl" x="{label_w - 10}" y="{y + bar_h - 3}" text-anchor="end">{_e(r["label"])}</text>'
                   f'<path d="{_bar_path(label_w, y, w, bar_h)}" fill="var({r["color"]})"/>'
                   f'<text class="val" x="{label_w + w + 6:.1f}" y="{y + bar_h - 3}">{_e(fmt(r["value"]))}</text></g>')
    out.append("</svg>")
    return "".join(out)


def _roc_svg(roc: dict, op: dict) -> str:
    W, H, x0, y0, w, h = 440, 380, 48, 14, 370, 320
    sx = lambda v: x0 + v * w  # noqa: E731
    sy = lambda v: y0 + h - v * h  # noqa: E731
    out = [f'<svg class="roc" viewBox="0 0 {W} {H}" width="{W}" role="img" '
           f'aria-label="ROC curve" data-roc="{_e(json.dumps(roc))}" data-x0="{x0}" data-y0="{y0}" '
           f'data-w="{w}" data-h="{h}">']
    for t in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{sx(t)}" x2="{sx(t)}" y1="{y0}" y2="{y0 + h}" stroke="var(--grid)"/>')
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{sy(t)}" y2="{sy(t)}" stroke="var(--grid)"/>')
        out.append(f'<text x="{sx(t)}" y="{y0 + h + 16}" text-anchor="middle">{int(t * 100)}%</text>')
        out.append(f'<text x="{x0 - 8}" y="{sy(t) + 4}" text-anchor="end">{int(t * 100)}%</text>')
    out.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="var(--axis)"/>')
    out.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="var(--axis)"/>')
    out.append(f'<text x="{sx(.62)}" y="{sy(.56)}" transform="rotate(-41 {sx(.62)} {sy(.56)})">chance</text>')
    out.append(f'<text x="{x0 + w / 2}" y="{H - 2}" text-anchor="middle">False-alarm rate</text>')
    out.append(f'<text x="12" y="{y0 + h / 2}" text-anchor="middle" transform="rotate(-90 12 {y0 + h / 2})">'
               f'Detection rate</text>')
    if roc["fpr"]:
        pts = " ".join(f"{sx(f):.1f},{sy(t):.1f}" for f, t in zip(roc["fpr"], roc["tpr"]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="var(--series-1)" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
    ox, oy = sx(op["false_alarm_rate"]), sy(op["detection_rate"])
    out.append(f'<line class="cross" x1="0" x2="0" y1="{y0}" y2="{y0 + h}" stroke="var(--axis)" style="display:none"/>')
    out.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="5" fill="var(--series-1)" stroke="var(--surface-1)" '
               f'stroke-width="2"/>')
    anchor = "start" if op["false_alarm_rate"] < .6 else "end"
    dx = 10 if anchor == "start" else -10
    out.append(f'<text class="lbl" x="{ox + dx:.1f}" y="{oy + 16:.1f}" text-anchor="{anchor}">'
               f'operating point: {_pct(op["detection_rate"])} detected at {_pct(op["false_alarm_rate"])} FAR</text>')
    out.append(f'<circle class="cross-dot" r="5" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2" '
               f'style="display:none"/>')
    out.append(f'<rect class="roc-hit" x="{x0}" y="{y0}" width="{w}" height="{h}" fill="transparent"/>')
    out.append("</svg>")
    return "".join(out)


def _confusion(mc: dict) -> str:
    labels, cm = mc["labels"], mc["confusion"]
    keep = [i for i, lab in enumerate(labels) if sum(cm[i]) > 0 or sum(r[i] for r in cm) > 0]
    head = "".join(f'<th class="rot">{_e(labels[j])}</th>' for j in keep)
    rows = []
    for i in keep:
        total = sum(cm[i]) or 1
        cells = []
        for j in keep:
            v = cm[i][j]
            share = v / total
            pct = round(share * 100)
            ink = "#fff" if pct >= 55 else "var(--text-primary)"
            cells.append(f'<td class="cell" style="background:color-mix(in oklab,var(--series-1) {pct}%,var(--surface-1));'
                         f'color:{ink}" data-tip="{_e(f"true {labels[i]} -> predicted {labels[j]}")}&#10;'
                         f'{v:,} rows ({share:.1%} of {labels[i]})">{share:.0%}</td>')
        rows.append(f'<tr><th>{_e(labels[i])}</th>{"".join(cells)}<td>{sum(cm[i]):,}</td></tr>')
    return (f'<div class="scroll"><table class="cm"><tr><th>true ↓ / predicted →</th>{head}'
            f'<th>rows</th></tr>{"".join(rows)}</table></div>')


def render_report(m: dict) -> str:
    meta, b, b5, mc = m["meta"], m["binary"], m["binary_at_0.5"], m["multiclass"]
    snv = m.get("seen_vs_novel")
    val = m.get("validation")
    ds = {"nsl-kdd": "NSL-KDD (KDDTrain+ → KDDTest+)", "cicids2017": "CICIDS2017 (stratified split)"}
    title = f"NetSentry · {meta['model']} on {meta['dataset']}"

    lede = (f"On {m['n']:,} held-out test records ({m['n_attack']:,} attacks), the detector flags "
            f"<b>{_pct(b['detection_rate'])}</b> of attacks while raising a false alarm on "
            f"<b>{_pct(b['false_alarm_rate'])}</b> of normal traffic.")
    if snv and snv["novel"]["n"]:
        lede += (f" On the {snv['novel']['attack_types']} attack types it <i>never saw during training</i> "
                 f"({snv['novel']['n']:,} records) it still catches <b>{_pct(snv['novel']['detection_rate'])}</b>.")
    if val and val["false_alarm_rate"] + 0.005 < b["false_alarm_rate"]:
        lede += (f" The threshold was tuned for {_pct(meta.get('target_far'))} false alarms on validation data "
                 f"(achieved {_pct(val['false_alarm_rate'])}); the higher rate on the test set reflects the "
                 f"distribution shift between the two.")

    tiles = [
        _tile("Detection rate", _pct(b["detection_rate"]), "attacks flagged (recall)"),
        _tile("False-alarm rate", _pct(b["false_alarm_rate"]), "normal traffic flagged"),
        _tile("ROC-AUC", f"{b['roc_auc']:.3f}" if b["roc_auc"] is not None else "n/a", "threshold-free ranking quality"),
        _tile("Family accuracy", _pct(mc["accuracy"]), f"macro-F1 {mc['macro_f1']:.3f} across families"),
    ]
    if snv and snv["novel"]["n"]:
        tiles.append(_tile("Novel attacks caught", _pct(snv["novel"]["detection_rate"]),
                           f"{snv['novel']['attack_types']} types unseen in training"))

    # Detection rate per attack type
    per_attack = m["per_attack"]
    has_seen = any(a["seen"] is not None for a in per_attack)
    bars = [{
        "label": a["attack"],
        "value": a["detection_rate"],
        "color": "--series-1" if a["seen"] is not False else "--series-2",
        "tip": (f"{a['attack']} ({a['category']})\n{a['n']:,} test records\n"
                f"detected {_pct(a['detection_rate'])}\nright family {_pct(a['family_accuracy'])}"
                + ("" if a["seen"] is None else ("\nseen in training" if a["seen"] else "\nNOT in training data"))),
    } for a in per_attack]
    legend = ('<div class="legend"><span><span class="key" style="background:var(--series-1)"></span>seen in training</span>'
              '<span><span class="key" style="background:var(--series-2)"></span>novel (not in training data)</span></div>'
              if has_seen else "")

    # Per-class table
    pc_rows = "".join(
        f"<tr><td>{_e(k)}</td><td>{v['precision']:.3f}</td><td>{v['recall']:.3f}</td><td>{v['f1']:.3f}</td>"
        f"<td>{v['support']:,}</td></tr>" for k, v in mc["per_class"].items() if v["support"] or v["precision"])

    def op_row(name, x):
        return (f"<tr><td>{_e(name)}</td><td>{x['threshold']:.4g}</td><td>{_pct(x['detection_rate'])}</td>"
                f"<td>{_pct(x['false_alarm_rate'])}</td><td>{_pct(x['precision'])}</td><td>{x['f1']:.3f}</td>"
                f"<td>{_pct(x['accuracy'])}</td></tr>")
    ops = op_row("Calibrated threshold (test)", b) + op_row("Default 0.5 (test)", b5)
    if val:
        ops += op_row("Calibrated threshold (validation)", val)

    fi = m.get("feature_importance") or []
    fi_html = ""
    if fi:
        top = fi[:15]
        mx = top[0][1] or 1
        fi_bars = [{"label": k, "value": v / mx, "color": "--series-1", "tip": f"{k}\n{v:.1%} of total importance"}
                   for k, v in top]
        fi_html = (f'<div class="card"><h2>What the model looks at</h2><p class="sub">Top input features by '
                   f'importance, scaled to the strongest feature. One-hot columns are summed back to their source '
                   f'field. Hover a bar for its share of total importance.</p>'
                   f'<div class="scroll">{_hbars(fi_bars, label_w=230, fmt=lambda v: "")}</div></div>')

    attack_table = "".join(
        f"<tr><td>{_e(a['attack'])}</td><td>{_e(a['category'])}</td><td>{a['n']:,}</td>"
        f"<td>{_pct(a['detection_rate'])}</td><td>{_pct(a['family_accuracy'])}</td>"
        f"<td>{'' if a['seen'] is None else ('yes' if a['seen'] else 'no')}</td></tr>" for a in per_attack)

    body = f"""
<main>
  <h1>{_e(title)}</h1>
  <p class="sub">{_e(ds.get(meta['dataset'], meta['dataset']))} · trained {_e(meta['trained_at'][:16].replace('T', ' '))} UTC ·
    {meta['n_fit']:,} training rows · fit in {meta['fit_seconds']}s ·
    scores ~{m.get('predict_rows_per_second', 0):,} rows/s</p>
  <p class="lede">{lede}</p>
  <div class="tiles">{''.join(tiles)}</div>

  <div class="grid2">
    <div class="card"><h2>ROC curve</h2>
      <p class="sub">Every possible alert threshold on the test set. The dot is the threshold this model ships with
      ({b['threshold']:.4g}). Hover the curve for other operating points.</p>
      {_roc_svg(m['roc'], b)}</div>
    <div class="card"><h2>Which family does it predict?</h2>
      <p class="sub">Row-normalised confusion matrix: each row is a true family and sums to 100%. Hover a cell for counts.</p>
      {_confusion(mc)}
      <h2 style="margin-top:20px">Per family</h2>
      <div class="scroll"><table><tr><th>family</th><th>precision</th><th>recall</th><th>F1</th><th>support</th></tr>
      {pc_rows}</table></div></div>
  </div>
  <div style="height:20px"></div>

  <div class="card"><h2>Detection rate by attack type</h2>
    <p class="sub">Share of each attack type's test records that raised an alert, largest types first.</p>
    {legend}<div class="scroll">{_hbars(bars)}</div>
    <details><summary>Table view</summary><div class="scroll"><table>
      <tr><th>attack</th><th>family</th><th>records</th><th>detected</th><th>right family</th><th>seen in training</th></tr>
      {attack_table}</table></div></details>
  </div>

  <div class="card"><h2>Operating points</h2>
    <p class="sub">The same model at its calibrated threshold vs. the naive 0.5 cut-off. Precision and accuracy
    depend on the attack/normal mix of the test set; detection and false-alarm rates do not.</p>
    <div class="scroll"><table><tr><th></th><th>threshold</th><th>detection</th><th>false alarms</th>
    <th>precision</th><th>F1</th><th>accuracy</th></tr>{ops}</table></div></div>

  {fi_html}

  <p class="foot">NetSentry {_e(meta['netsentry_version'])} · scikit-learn {_e(meta['sklearn_version'])} ·
  model <code>{_e(meta['model'])}</code>{' (class-balanced)' if meta.get('balanced') else ''} · seed {meta['seed']} ·
  validation rows {meta['n_val']:,} · test rows {meta['n_test']:,}</p>
</main>
<div id="tip" hidden></div>"""

    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>NetSentry report</title><style>{CSS}</style></head><body>{body}'
            f'<script>{JS}</script></body></html>')


def write_report(metrics: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
    return path


def write_metrics(metrics: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
