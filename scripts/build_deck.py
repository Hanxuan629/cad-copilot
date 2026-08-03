"""Self-contained HTML slide-deck builder — figure-forward ("少说话多画图").

Copy this file into your project's scripts/, rename it, and edit the SLIDES
list + ROOT/OUT paths below. Each content slide = one hero figure + a one-line
kicker (the claim) + a few rail bullets, so the figure carries the talk.

Figures are base64-embedded → the single .html is fully portable (email/move
without broken paths). Dark 16:9 theme, arrow-key / click nav, print-to-PDF.

Run:
  python scripts/build_<name>_deck.py
Output:
  results/<name>_deck.html
"""

from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path

# --- EDIT THESE -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # repo root
OUT = ROOT / "results" / "cad_copilot_deck.html"    # output file
DECK_LANG = "en"                                     # "zh" or "en"
DECK_TITLE_TAB = "CAD-Copilot"                       # browser tab title
# ---------------------------------------------------------------------------


def embed(rel_path: str) -> str:
    """Return a data: URI for an image, or '' (and a warning) if missing."""
    p = ROOT / rel_path
    if not p.exists():
        print(f"  !! MISSING {rel_path}")
        return ""
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def esc(s: str) -> str:
    return html.escape(str(s))


# --------------------------------------------------------------------------- #
# Example native diagram (SVG). Prefer native over raster for diagrams /
# equations / tables — crisp and never goes missing. If you add a new native
# block, give it a class and add that class to the `hasnative` JS selector.
# --------------------------------------------------------------------------- #
SPLIT_NATIVE_HTML = """
<div class="mech">
  <div class="mech-eq">sketch &rarr; <b>VLM (perception)</b> &rarr; <b>symbolic solver (geometry)</b> &rarr; CAD</div>
  <svg viewBox="0 0 620 200" preserveAspectRatio="xMidYMid meet">
    <rect x="20"  y="70" width="120" height="60" rx="6" fill="none" stroke="#5b6472" stroke-width="2"/>
    <text x="80" y="105" fill="#e6edf3" font-size="13" text-anchor="middle">sketch PNG</text>
    <rect x="190" y="70" width="130" height="60" rx="6" fill="none" stroke="#58a6ff" stroke-width="3"/>
    <text x="255" y="97" fill="#58a6ff" font-size="13" font-weight="700" text-anchor="middle">VLM (LoRA)</text>
    <text x="255" y="116" fill="#8b949e" font-size="11" text-anchor="middle">what to draw</text>
    <rect x="370" y="70" width="140" height="60" rx="6" fill="none" stroke="#f0883e" stroke-width="3"/>
    <text x="440" y="97" fill="#f0883e" font-size="13" font-weight="700" text-anchor="middle">symbolic solver</text>
    <text x="440" y="116" fill="#8b949e" font-size="11" text-anchor="middle">precisely where</text>
    <rect x="560" y="70" width="50" height="60" rx="6" fill="none" stroke="#5b6472" stroke-width="2"/>
    <text x="585" y="105" fill="#e6edf3" font-size="12" text-anchor="middle">CAD</text>
    <line x1="140" y1="100" x2="188" y2="100" stroke="#8b949e" stroke-width="2" marker-end="url(#a)"/>
    <line x1="320" y1="100" x2="368" y2="100" stroke="#8b949e" stroke-width="2" marker-end="url(#a)"/>
    <line x1="510" y1="100" x2="558" y2="100" stroke="#8b949e" stroke-width="2" marker-end="url(#a)"/>
    <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6" fill="#8b949e"/></marker></defs>
  </svg>
  <div class="mech-ops">
    <div class="op add"><span class="dot"></span><b>perception</b>
      <div>primitive type, count, rough place &mdash; the VLM is good at this</div></div>
    <div class="op mod"><span class="dot"></span><b>geometry</b>
      <div>exact coordinates &mdash; snap to image ink by least squares</div></div>
    <div class="op del"><span class="dot"></span><b>boundary</b>
      <div>split the task where each tool is strong</div></div>
  </div>
</div>
"""


# --------------------------------------------------------------------------- #
# Slide content — EDIT THIS. See SKILL.md for the full kind reference.
# --------------------------------------------------------------------------- #
SLIDES: list[dict] = [
    {
        "kind": "cover",
        "title": "CAD-Copilot",
        "subtitle": "Neuro-symbolic CAD sketch parsing, and where a VLM's ability ends",
        "bullets": [
            "① End-to-end sketch→CAD is bottlenecked by perception",
            "② Split the task: VLM does perception, symbolic solver does geometry",
            "③ Across 3 probes: big models are strong at perception, weak at precise geometry",
        ],
    },

    {
        "kind": "section", "sec": "1",
        "title": "The problem",
        "note": "A stock VLM cannot read a CAD sketch precisely enough to reconstruct it",
    },
    {
        "kind": "figure", "sec": "1",
        "kicker": "One sketch, three views: the VLM misreads structure and geometry",
        "title": "Failure case",
        "img": "results/example_fc0f969bc46c40b6780eb8ff_0_3.png",
        "caption": "GT (green) vs LoRA prediction (red) vs after symbolic refine (blue), over the true sketch",
        "points": [
            "Target = irregular quad + arc notch + 2 holes",
            "LoRA draws a regular rectangle + 6 stray circles — misses the arc, over-generates",
            "→ end-to-end reconstruction is limited by perception, not by the solver",
        ],
    },

    {
        "kind": "section", "sec": "2",
        "title": "The method",
        "note": "Decompose along the capability boundary",
    },
    {
        "kind": "figure", "sec": "2", "native": SPLIT_NATIVE_HTML,
        "kicker": "VLM owns “what to draw”; a symbolic solver owns “precisely where”",
        "title": "Neuro-symbolic split",
        "caption": "rendered natively — the pipeline, not an image",
        "points": [
            "Perception (type / count / rough place) → LoRA-finetuned Qwen3-VL-2B",
            "Geometry (exact coordinates) → least-squares snap to image ink, PNG-only",
            "→ measure each axis separately: edit-distance vs chamfer",
        ],
    },
    {
        "kind": "figure", "sec": "2",
        "kicker": "LoRA fixes perception; only the symbolic stage fixes geometry",
        "title": "Stage 1 + 2 results",
        "img": "results/figs/stages.png",
        "caption": "eval_model.py + refine.py · 500-image eval (base/LoRA), 200-image subset (refine)",
        "points": [
            "type accuracy 0.2% → 39% with LoRA (perception jumps)",
            "chamfer flat under LoRA (2.90 → 2.94); drops to 2.48 only after symbolic refine",
            "→ the two gains are orthogonal — the boundary is real",
        ],
    },

    {
        "kind": "section", "sec": "3",
        "title": "Discrete sub-tasks",
        "note": "Where does the VLM actually help? Two clean probes on MRCAD data",
    },
    {
        "kind": "figure", "sec": "3",
        "kicker": "Task 2 setup: pair each drawn curve to the target curve it represents",
        "title": "What is line matching?",
        "img": "results/figs/task_setup.png",
        "caption": "one real MRCAD trial · left = human-drawn CAD, right = target · same colour = a matched pair",
        "points": [
            "Input: a set of drawn curves + a set of target curves (same coordinate frame)",
            "Output: for each drawn curve, which target it corresponds to (one-to-many allowed)",
            "→ a discrete correspondence problem — no coordinate generation needed",
        ],
    },
    {
        "kind": "figure", "sec": "3",
        "kicker": "Line matching: the VLM is near-chance; training-free geometry is exact",
        "title": "Task 2 — result",
        "img": "results/figs/matching.png",
        "caption": "probe_match.py + eval_probe.py · 50-trial eval, top-1 accuracy vs geometric GT",
        "points": [
            "For each drawn curve: which target curve does it match?",
            "Geometric chamfer-NN = 1.00; zero-shot VLM = 0.15 (text) / 0.11 (visual)",
            "→ a discrete geometric-scoring task — the VLM is the wrong tool",
        ],
    },
    {
        "kind": "figure", "sec": "3",
        "kicker": "A text embedding misses geometric similarity; a small geo-AE captures it",
        "title": "Task 3 — similarity embedding",
        "img": "results/figs/embedding.png",
        "caption": "cad_autoencoder.py + embed_compare.py · Spearman ρ vs chamfer distance, 3000 pairs",
        "points": [
            "Set-autoencoder over a design's curves (~256-d, reconstruction loss, val 0.179)",
            "geo-AE ρ = 0.41 vs Qwen3-Embedding on CAD-code text ρ = 0.04",
            "→ geometric similarity needs a geometry-native embedding, not a text one",
        ],
    },

    {
        "kind": "design",
        "kicker": "One consistent finding across every probe",
        "thesis": "Large pretrained VLM / text models are strong at perception and semantics but "
                  "weak at precise geometry. Geometry is better served by small, geometry-native "
                  "methods — least-squares refinement, geometric matching, a set-autoencoder. "
                  "The win is putting each tool where it is strong.",
        "levers": [
            {"tag": "Perception → neural",
             "why": "VLM lifts type accuracy 0.2%→39% but never fixes coordinates",
             "do": "keep the VLM for structure / semantics / language"},
            {"tag": "Geometry → symbolic",
             "why": "chamfer only drops with the solver; matching is exact without a model",
             "do": "least-squares snap, geometric matching, geo-native embeddings"},
            {"tag": "Next",
             "why": "task 1 (direct-draw vs annotation) still needs a supervision source",
             "do": "locate auxiliary-mark labels, then reuse the same probe harness"},
        ],
    },
]


# --------------------------------------------------------------------------- #
# HTML rendering (reusable — usually no need to edit below here)
# --------------------------------------------------------------------------- #
def render_slide(s: dict, idx: int, total: int) -> str:
    kind = s["kind"]
    foot = f'<div class="pageno">{idx} / {total}</div>'

    if kind == "cover":
        items = "".join(f"<li>{esc(b)}</li>" for b in s["bullets"])
        return f"""<section class="slide cover">
  <div class="cover-wrap">
    <div class="eyebrow">SYNC</div>
    <h1>{esc(s['title'])}</h1>
    <div class="subtitle">{esc(s['subtitle'])}</div>
    <ol class="agenda">{items}</ol>
  </div>
  {foot}
</section>"""

    if kind == "section":
        return f"""<section class="slide section">
  <div class="sec-badge">{esc(s['sec'])}</div>
  <div>
    <h1 class="sec-title">{esc(s['title'])}</h1>
    <div class="sec-note">{esc(s['note'])}</div>
  </div>
  {foot}
</section>"""

    if kind == "end":
        items = "".join(f"<li>{esc(b)}</li>" for b in s["bullets"])
        return f"""<section class="slide end">
  <div class="cover-wrap">
    <div class="eyebrow">NEXT</div>
    <h1>{esc(s['title'])}</h1>
    <ul class="next">{items}</ul>
  </div>
  {foot}
</section>"""

    if kind == "design":
        levers = "".join(
            f'<div class="dl-card"><div class="dl-tag">{esc(l["tag"])}</div>'
            f'<div class="dl-why"><span>WHY</span>{esc(l["why"])}</div>'
            f'<div class="dl-do"><span>DO</span>{esc(l["do"])}</div></div>'
            for l in s["levers"]
        )
        return f"""<section class="slide figslide">
  <header class="fig-hd">
    <div class="sec-chip">✦</div>
    <div class="kicker">{esc(s['kicker'])}</div>
  </header>
  <div class="design-body">
    <div class="dz-thesis">{esc(s['thesis'])}</div>
    <div class="dz-grid">{levers}</div>
  </div>
  {foot}
</section>"""

    if kind == "twocol":
        def col(c):
            lis = "".join(f"<li>{esc(x)}</li>" for x in c["points"])
            return (f'<div class="tc-card" style="border-top-color:{c["accent"]}">'
                    f'<div class="tc-head"><span class="tc-tag" '
                    f'style="background:{c["accent"]}">{esc(c["tag"])}</span>'
                    f'<h3>{esc(c["title"])}</h3></div>'
                    f'<div class="tc-sub">{esc(c.get("sub",""))}</div>'
                    f'<ul>{lis}</ul></div>')
        cols = "".join(col(c) for c in s["cols"])
        chip = f'<div class="sec-chip">{esc(s["sec"])}</div>' if s.get("sec") else ""
        return f"""<section class="slide figslide">
  <header class="fig-hd">
    {chip}
    <div class="kicker">{esc(s['kicker'])}</div>
  </header>
  <div class="twocol">{cols}</div>
  {foot}
</section>"""

    # figure slide
    pts = "".join(f"<li>{esc(p)}</li>" for p in s.get("points", []))
    if s.get("native"):
        figure = f'<div class="fig-frame">{s["native"]}</div>'
    elif s.get("imgs"):
        frames = "".join(
            f'<div class="fig-frame">'
            f'<img src="{embed(im)}" alt="{esc(s["title"])}"/></div>'
            for im in s["imgs"]
        )
        figure = f'<div class="fig-stack">{frames}</div>'
    else:
        uri = embed(s["img"])
        figure = f'<div class="fig-frame"><img src="{uri}" alt="{esc(s["title"])}"/></div>'
    cap = f'<div class="cap">{esc(s["caption"])}</div>' if s.get("caption") else ""
    return f"""<section class="slide figslide">
  <header class="fig-hd">
    <div class="sec-chip">{esc(s['sec'])}</div>
    <div class="kicker">{esc(s['kicker'])}</div>
  </header>
  <div class="fig-body">
    <div class="fig-col">
      {figure}
      {cap}
    </div>
    <aside class="rail">
      <h2>{esc(s['title'])}</h2>
      <ul>{pts}</ul>
    </aside>
  </div>
  {foot}
</section>"""


def build() -> str:
    total = len(SLIDES)
    slides_html = "\n".join(
        render_slide(s, i + 1, total) for i, s in enumerate(SLIDES)
    )
    return (TEMPLATE
            .replace("{{SLIDES}}", slides_html)
            .replace("{{TOTAL}}", str(total))
            .replace("{{LANG}}", DECK_LANG)
            .replace("{{TABTITLE}}", esc(DECK_TITLE_TAB)))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="{{LANG}}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TABTITLE}}</title>
<style>
  :root{
    --bg:#0f1115; --card:#171a21; --ink:#e9edf3; --muted:#9aa4b2;
    --line:#262b35; --add:#4C9BE6; --mod:#F2A63B; --del:#E05656;
    --ok:#4FBF8B; --accent:#ffcf33;
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  html,body{height:100%;}
  body{
    background:#05070b; color:var(--ink);
    font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
    display:flex; align-items:center; justify-content:center; min-height:100vh;
    overflow:hidden;
  }
  .deck{width:100vw; height:100vh; display:flex; align-items:center; justify-content:center;}
  .slide{
    position:absolute; width:min(1280px,96vw); aspect-ratio:16/9;
    background:linear-gradient(160deg,#141821,#0f1115);
    border:1px solid var(--line); border-radius:16px;
    padding:38px 46px 30px; box-shadow:0 30px 90px rgba(0,0,0,.6);
    display:none; flex-direction:column; overflow:hidden;
  }
  .slide.active{display:flex;}
  .pageno{position:absolute; right:20px; bottom:14px; color:var(--muted);
    font-size:12px; font-family:"SF Mono",ui-monospace,Menlo,monospace;}

  /* cover */
  .cover, .end{justify-content:center;}
  .cover-wrap{display:flex; flex-direction:column; gap:14px;}
  .eyebrow{color:var(--accent); font-size:13px; letter-spacing:2px; font-weight:700;}
  .cover h1, .end h1{font-size:44px; line-height:1.15; font-weight:800; letter-spacing:.5px;}
  .subtitle{color:var(--muted); font-size:19px; margin-bottom:8px;}
  .agenda{list-style:none; display:grid; grid-template-columns:1fr 1fr; gap:10px 30px; margin-top:6px;}
  .agenda li{font-size:17px; color:#cfd6e0; padding:9px 14px; background:var(--card);
    border:1px solid var(--line); border-radius:10px;}

  /* section divider */
  .section{flex-direction:row; align-items:center; gap:34px; justify-content:flex-start;}
  .sec-badge{font-size:120px; font-weight:800; line-height:1;
    color:#232a36; -webkit-text-stroke:2px var(--accent);}
  .sec-title{font-size:40px; font-weight:800;}
  .sec-note{color:var(--muted); font-size:19px; margin-top:12px;}

  /* figure slide */
  .figslide{gap:14px;}
  .fig-hd{display:flex; align-items:center; gap:14px; border-bottom:1px solid var(--line);
    padding-bottom:14px;}
  .sec-chip{flex:none; width:30px; height:30px; border-radius:8px; background:#222834;
    color:var(--accent); font-weight:800; display:grid; place-items:center; font-size:15px;}
  .kicker{font-size:23px; font-weight:750; line-height:1.25; letter-spacing:.2px;}
  .fig-body{display:grid; grid-template-columns:1.62fr 1fr; gap:26px; flex:1; min-height:0;}
  .fig-col{display:flex; flex-direction:column; min-height:0; gap:8px;}
  .fig-frame{flex:1; min-height:0; background:#fff; border:1px solid var(--line);
    border-radius:12px; padding:12px; display:flex; align-items:center; justify-content:center;}
  .fig-frame img{max-width:100%; max-height:100%; object-fit:contain;}
  .fig-stack{flex:1; min-height:0; display:flex; flex-direction:column; gap:8px;}
  .fig-stack .fig-frame{flex:1; padding:8px;}
  .cap{color:var(--muted); font-size:12px; font-family:"SF Mono",ui-monospace,Menlo,monospace;}
  .rail{display:flex; flex-direction:column; min-height:0;}
  .rail h2{font-size:18px; color:var(--ink); margin-bottom:12px; font-weight:750;
    padding-left:12px; border-left:3px solid var(--accent);}
  .rail ul{list-style:none; display:flex; flex-direction:column; gap:11px; overflow:auto;}
  .rail li{font-size:15px; color:#d0d7e1; line-height:1.45; padding-left:18px; position:relative;}
  .rail li::before{content:"▸"; color:var(--accent); position:absolute; left:0; opacity:.85;}

  /* two-column comparison slide */
  .twocol{display:grid; grid-template-columns:1fr 1fr; gap:24px; flex:1; min-height:0; margin-top:6px;}
  .tc-card{background:var(--card); border:1px solid var(--line); border-top:3px solid var(--accent);
    border-radius:12px; padding:20px 22px; display:flex; flex-direction:column; min-height:0; overflow:auto;}
  .tc-head{display:flex; align-items:center; gap:12px; margin-bottom:4px;}
  .tc-tag{color:#0d0f14; font-size:12px; font-weight:800; padding:3px 10px; border-radius:20px; letter-spacing:.5px;}
  .tc-head h3{font-size:21px; font-weight:750;}
  .tc-sub{color:var(--muted); font-size:13.5px; margin-bottom:14px;}
  .tc-card ul{list-style:none; display:flex; flex-direction:column; gap:13px;}
  .tc-card li{font-size:15.5px; color:#d5dce6; line-height:1.5; padding-left:20px; position:relative;}
  .tc-card li::before{content:"•"; position:absolute; left:2px; color:var(--muted); font-size:18px; line-height:1.2;}

  /* experiment-design finale slide */
  .design-body{display:flex; flex-direction:column; gap:16px; flex:1; min-height:0; margin-top:6px;}
  .dz-thesis{background:#12161d; border:1px solid var(--line); border-left:4px solid var(--accent);
    border-radius:12px; padding:16px 20px; font-size:16.5px; line-height:1.6; color:#e2e8f1;}
  .dz-grid{display:grid; grid-template-columns:1fr 1fr; gap:14px; flex:1; min-height:0;}
  .dl-card{background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:14px 18px; display:flex; flex-direction:column; gap:8px; overflow:auto;}
  .dl-tag{font-size:16px; font-weight:750; color:var(--accent);}
  .dl-why, .dl-do{font-size:14px; line-height:1.5; color:#d0d7e1; padding-left:52px; position:relative;}
  .dl-why span, .dl-do span{position:absolute; left:0; top:0; font-size:11px; font-weight:700;
    letter-spacing:.5px; padding:2px 7px; border-radius:6px;}
  .dl-why span{color:var(--muted); background:#0e1219; border:1px solid var(--line);}
  .dl-do span{color:#0d0f14; background:var(--ok);}
  .dl-why{color:var(--muted);}

  /* native mechanics figure (dark, not white frame) */
  .figslide.hasnative .fig-frame{background:#0e1219;}
  .mech{width:100%; height:100%; display:flex; flex-direction:column; gap:10px; color:var(--ink);}
  .mech-eq{font-size:14px; color:var(--muted); font-family:"SF Mono",ui-monospace,Menlo,monospace; text-align:center;}
  .mech svg{flex:1; min-height:0; width:100%;}
  .mech-ops{display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;}
  .mech .op{border:1px solid var(--line); border-radius:9px; padding:8px 10px; background:#12161d; font-size:11.5px;}
  .mech .op b{font-size:13px;} .mech .op div{color:var(--muted); margin-top:4px; line-height:1.35;}
  .mech .op .dot{width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:6px;}
  .mech .add b{color:var(--add);} .mech .add .dot{background:var(--add);}
  .mech .mod b{color:var(--mod);} .mech .mod .dot{background:var(--mod);}
  .mech .del b{color:var(--del);} .mech .del .dot{background:var(--del);} .mech .del u{color:var(--del);}

  /* end */
  .next{list-style:none; display:flex; flex-direction:column; gap:12px; margin-top:8px;}
  .next li{font-size:19px; color:#cfd6e0; padding:12px 16px; background:var(--card);
    border:1px solid var(--line); border-left:3px solid var(--ok); border-radius:10px;}

  /* progress + hint */
  .progress{position:fixed; top:0; left:0; height:3px; background:var(--accent); z-index:10; transition:width .2s;}
  .hint{position:fixed; left:16px; bottom:12px; color:#404a58; font-size:12px; z-index:10;}

  @media print{
    body{overflow:visible;} .deck{display:block;}
    .slide{display:flex !important; position:relative; page-break-after:always;
      width:100%; aspect-ratio:16/9; margin:0 auto; box-shadow:none; border-radius:0;}
    .progress,.hint{display:none;}
  }
</style>
</head>
<body>
  <div class="progress" id="prog"></div>
  <div class="deck" id="deck">
    {{SLIDES}}
  </div>
  <div class="hint">← → / Space to navigate · F fullscreen · P print to PDF</div>
<script>
  const slides=[...document.querySelectorAll('.slide')];
  // tag native-figure slides so their frame goes dark (add new native classes here)
  slides.forEach(s=>{ if(s.querySelector('.mech, .hyp, .sdm')) s.classList.add('hasnative'); });
  let i=0;
  const prog=document.getElementById('prog');
  function show(n){
    i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach((s,k)=>s.classList.toggle('active',k===i));
    prog.style.width=((i)/(slides.length-1)*100)+'%';
    location.hash=i+1;
  }
  function next(){show(i+1);} function prev(){show(i-1);}
  document.addEventListener('keydown',e=>{
    if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();next();}
    else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();prev();}
    else if(e.key==='Home'){show(0);} else if(e.key==='End'){show(slides.length-1);}
    else if(e.key==='f'||e.key==='F'){if(!document.fullscreenElement)document.documentElement.requestFullscreen();else document.exitFullscreen();}
    else if(e.key==='p'||e.key==='P'){window.print();}
  });
  document.getElementById('deck').addEventListener('click',e=>{
    if(e.clientX < window.innerWidth*0.28) prev(); else next();
  });
  const start=parseInt(location.hash.slice(1)); show(isNaN(start)?0:start-1);
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("Building deck …")
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    kb = len(doc.encode("utf-8")) / 1024
    print(f"→ {OUT}  ({kb:.0f} KB, {len(SLIDES)} slides)")
