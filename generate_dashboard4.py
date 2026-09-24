#!/usr/bin/env python3
"""
Build the three-class dashboard (v4) from three_class_results.json.
Preserves the sentiment/emotion analysis and the review explorer, now for THREE
sentiment classes (positive / neutral / negative) on a balanced 150-review sample.
Single self-contained offline HTML.
"""
import json, html

SRC = "three_class_results.json"
OUT = "Gift_Card_Classifier_Dashboard_v5.html"

d   = json.load(open(SRC, encoding="utf-8"))
rows = d["rows"]
n   = d["sampling"]["total"]
CLS_ORDER = ["positive", "neutral", "negative"]
CLS = d["sentiment_overall"]
by_cls = d["class_accuracy"]
EMOT = d["emotion_agreement"]
EMO   = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

agree = CLS["agree"]; pct = CLS["agree_pct"]; miss = CLS["miss"]
e_agree = EMOT["agree"]; e_pct = EMOT["agree_pct_all"]; e_none = EMOT["nrc_none"]; e_dis = EMOT["disagree"]
model_dist = d["model_emotion_distribution"]; nrc_dist = d["nrc_emotion_distribution"]
per_cls = list(d["sampling"]["classes"].values())[0]
seed_val = d["sampling"]["seed"]

# categories to chart = any appearing in either distribution (EMOTIONS first, then extras, then none)
chart_order = [e for e in EMO if e in model_dist or e in nrc_dist]
for k in list(model_dist) + list(nrc_dist):
    if k not in chart_order and k != "none":
        chart_order.append(k)
if "none" in model_dist or "none" in nrc_dist:
    chart_order.append("none")
model_series = [model_dist.get(k, 0) for k in chart_order]
nrc_series   = [nrc_dist.get(k, 0) for k in chart_order]

def esc(t): return html.escape(t or "").replace('"', "&quot;")

def cls2key(c): return {"positive": "ok", "neutral": "neu", "negative": "bad"}[c]
def cls_lab(c): return {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}[c]
def cls_note(c, b):
    parts = []
    for k in CLS_ORDER:
        if k != c and b["pred_distribution"].get(k, 0) > 0:
            parts.append("%d read as %s" % (b["pred_distribution"][k], k))
    return ("Correct labels are amber/green/red by class; the grey shares are reviews the model "
            "called a different class. " + "; ".join(parts)) if parts else "All %d matched." % b["n"]

# ---- per-class alignment bars (prediction split) ----
class_bars = []
for c in CLS_ORDER:
    b = by_cls[c]
    pd = b["pred_distribution"]
    segs = "".join(
        '<div class="pseg seg-%s" style="width:%.2f%%"></div>' % (cls2key(k), 100.0 * pd.get(k, 0) / b["n"])
        for k in CLS_ORDER)
    class_bars.append(f"""
      <div class="cbar">
        <div class="row"><span class="clab"><i class="dot dot-{c}"></i>{c.title()}
          <span style="color:var(--faint);font-weight:400">({b["n"]} reviews)</span></span>
          <span class="cval">{b["correct"]} of {b["n"]} right &nbsp;·&nbsp; {b["acc"]}%</span></div>
        <div class="track">{segs}</div>
        <div class="track-note">{cls_note(c, b)}</div>
      </div>""")
class_bars_html = "\n".join(class_bars)

# ---- representative sentiment misses (a few from each class) ----
miss_rows = [r for r in rows if not r["correct"]]
# pick up to 9, aiming 3 per class
miss_sample = []
for c in CLS_ORDER:
    for r in miss_rows:
        if r["cls"] == c:
            miss_sample.append(r)
    # cap 3 per class below
per_class_cap = []
for c in CLS_ORDER:
    per_class_cap += [r for r in miss_rows if r["cls"] == c][:3]
miss_sample = per_class_cap

def cls_lab(c): return {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}[c]

miss_cards = []
for r in miss_sample:
    stars = "★" * int(round(r["rating"]))
    miss_cards.append(f"""
      <article class="miss">
        <div class="miss-rail"></div>
        <div class="miss-body">
          <div class="miss-meta">
            <span class="stars">{stars}</span>
            <span class="chip {cls2key(r['cls'])}">{cls_lab(r['cls'])}</span><span class="vs">rating says</span>
            <span class="arrow">&rarr;</span>
            <span class="chip {cls2key(r['sentiment'])}">{cls_lab(r['sentiment'])}</span><span class="vs">model read</span>
          </div>
          <h3 class="miss-title">{esc(r["title"])}</h3>
          <p class="miss-text">{esc(r["text"])}</p>
          <p class="miss-why"><b>Rating says {cls_lab(r['cls'])} ({"★"*int(round(r["rating"]))})</b>, but the model read the
             title+text as <b>{cls_lab(r['sentiment'])}</b>.</p>
        </div>
      </article>""")
miss_cards_html = "\n".join(miss_cards)

# ---- emotion disagreement examples ----
seen = set(); diffs = []
for r in rows:
    m, nr = r["model_emotion"], r["nrc_emotion"]
    if m != nr and m in EMO and nr in EMO:
        if (m, nr) in seen:
            continue
        seen.add((m, nr)); diffs.append(r)
    if len(diffs) >= 7:
        break
def nrc_words(r, limit=10):
    words = r.get("nrc_matched", {}).get(r["nrc_emotion"], []) if r["nrc_emotion"] in EMO else []
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
    return ", ".join(uniq[:limit]) or "(none found)"
diff_cards = []
for r in diffs:
    m, nr = r["model_emotion"], r["nrc_emotion"]
    diff_cards.append(f"""
      <article class="diff">
        <div class="diff-top">
          <span class="stars">{"★"*int(round(r["rating"]))}</span>
          <span class="diff-title">{esc(r["title"])}</span>
          <span class="diff-sent">{r["cls"]}</span>
        </div>
        <div class="diff-mid">
          <div class="emo-box"><span class="emo-lab">Model</span><span class="chip nl">{m}</span></div>
          <span class="arrow">&harr;</span>
          <div class="emo-box"><span class="emo-lab">NRC word-list</span><span class="chip nr">{nr}</span></div>
        </div>
        <p class="diff-why">The word-list picked <b>{nr}</b> from the words it counted there:
           <i>{esc(nrc_words(r))}</i>. The model instead read the review's emotion as <b>{m}</b>.</p>
      </article>""")
diff_cards_html = "\n".join(diff_cards)

ROWS_JSON = json.dumps(rows, ensure_ascii=False)

# ===================== descriptive layer (at-a-glance) =====================
from collections import Counter
TRUE  = Counter(r["cls"] for r in rows)            # rating-based ground truth
PRED  = Counter(r["sentiment"] for r in rows)      # model predictions
STAR  = Counter(int(round(r["rating"])) for r in rows)
CLSCOLOR = {"positive": "var(--ok)", "neutral": "var(--neu)", "negative": "var(--bad)"}
ORDER3 = ["positive", "neutral", "negative"]
max_star = max(STAR.values()) or 1
max_rp   = max(list(TRUE.values()) + list(PRED.values()))

# 1) star-rating distribution (vertical bars, heights scaled to max; zero-count omitted)
star_cols = []
for s in range(1, 6):
    c = STAR.get(s, 0)
    color = CLSCOLOR["positive"] if s >= 4 else (CLSCOLOR["neutral"] if s == 3 else CLSCOLOR["negative"])
    h = max(100.0 * c / max_star, 5.0) if c > 0 else 0.0
    star_cols.append(
        '<div class="vcol"><div class="vval">%d</div>'
        '<div class="vbar"><div class="vfill" style="height:%.1f%s;background:%s"></div></div>'
        '<div class="vlab">%d&#9733;</div></div>' % (c, h, "%", color, s))
stars_html = "".join(star_cols)

# 2) rating says (true) vs model predicted, grouped per class
rp_groups = []
for c in ORDER3:
    tru, pr = TRUE[c], PRED[c]
    h_tru = max(100.0 * tru / max_rp, 5.0) if tru > 0 else 0.0
    h_pr  = max(100.0 * pr / max_rp, 5.0) if pr > 0 else 0.0
    rp_groups.append(
        '<div class="rp-group">'
        '<div class="rp-head"><i class="dot dot-%s"></i>%s <span class="rp-nums">rating %d &middot; model %d</span></div>'
        '<div class="rp-cols">'
        '<div class="rp-col"><div class="vvar">%d</div><div class="vbar"><div class="vfill" style="height:%.1f%s"></div></div><div class="vlab">rating</div></div>'
        '<div class="rp-col"><div class="vvar">%d</div><div class="vbar"><div class="vfill" style="height:%.1f%s;background:%s"></div></div><div class="vlab">model</div></div>'
        '</div></div>'
        % (c, c.title(), tru, pr, tru, h_tru, "%", pr, h_pr, "%", CLSCOLOR[c]))
rp_html = "".join(rp_groups)

# 3) how often each class was answered right (horizontal bars)
acc_rows = []
for c in ORDER3:
    b = by_cls[c]
    fill = 100.0 * b["correct"] / b["n"]
    acc_rows.append(
        '<div class="abar-row">'
        '<div class="abar-lab"><span class="abar-name"><i class="dot dot-%s"></i>%s</span>'
        '<span class="abar-v">%s%% &middot; %d of %d right</span></div>'
        '<div class="abar"><div class="abar-fill" style="width:%.1f%s;background:%s"></div></div>'
        '</div>' % (c, c.title(), b["acc"], b["correct"], b["n"], fill, "%", CLSCOLOR[c]))
overall_fill = 100.0 * agree / n
acc_rows.append(
    '<div class="abar-row total"><div class="abar-lab"><span class="abar-name">Overall</span>'
    '<span class="abar-v">%d%% &middot; %d of %d right</span></div>'
    '<div class="abar"><div class="abar-fill" style="width:%.1f%s;background:var(--ink)"></div></div>'
    '</div>' % (pct, agree, n, overall_fill, "%"))
acc_html = "".join(acc_rows)

# biggest single failure type (for the at-a-glance callout)
fail_types = []
for c in ORDER3:
    for k in ORDER3:
        if k != c:
            cnt = by_cls[c]["pred_distribution"].get(k, 0)
            if cnt > 0:
                fail_types.append((cnt, "%s reviews read as %s" % (c, k)))
fail_types.sort(reverse=True)
top_fail = fail_types[0] if fail_types else (0, "")
top_fail_text = ("Most common miss: <b>%d</b> — %s (" % (top_fail[0], top_fail[1]) +
                 "%.0f%% of the %d misses)." % (100.0 * top_fail[0] / max(miss, 1), miss))

overview_html = f"""
<section id="overview">
  <p class="kicker">Descriptive layer &middot; at a glance</p>
  <h2>Where the model succeeds — and where it fails</h2>
  <p class="sub">
    No drilling needed: this shows the exact star mix we scored, how the model&rsquo;s verdicts
    compare with the rating-based answer, and how often each class is answered right. Green is the
    positive class, amber the neutral, red the negative. The big story: <b>the model over-predicts
    negative</b> and <b>stumbles hardest on neutral reviews</b>.
  </p>

  <div class="glance-callout">{top_fail_text} Everything else below matches what you saw in the
  individual rows.</div>

  <div class="ov-grid">
    <div class="ov-card ov-stars">
      <h4>Star-rating distribution</h4>
      <p class="cap">How the balanced sample is made up, by stars (coloured by class).</p>
      <div class="stars-chart" style="--max:{max_star}">{stars_html}</div>
    </div>

    <div class="ov-card ov-rp">
      <h4>Rating says vs. model predicted</h4>
      <p class="cap">True class from the rating (grey) against what the model called them (class colour), per class.</p>
      <div class="rp-chart">{rp_html}</div>
      <div class="rp-legend"><span><i class="sw" style="background:var(--line-strong)"></i>rating says</span>
        <span><i class="sw" style="background:var(--ok)"></i>model</span></div>
    </div>

    <div class="ov-card ov-acc">
      <h4>How often each class was answered right</h4>
      <p class="cap">Correct share of each class and overall.</p>
      <div class="acc-chart">{acc_html}</div>
    </div>
  </div>
</section>

"""

ORDER_JSON = json.dumps(chart_order)
SRC_NOTE = ("balanced sample of %d reviews (%d per rating class) drawn with a fixed random seed "
            "(%d) so the same set appears every time" % (n, list(d["sampling"]["classes"].values())[0], d["sampling"]["seed"]))

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gift Card Review Classifier — Three-Class Sentiment &amp; Emotion</title>
<style>
  :root {{
    --paper:#f5f4f0; --surface:#ffffff;
    --ink:#181c22; --muted:#5d6670; --faint:#8a9199;
    --line:#e2dfd8; --line-strong:#cfcbc1;
    --ok:#1f7a5c; --ok-soft:#e3f0eb;      /* positive */
    --neu:#a57a1d; --neu-soft:#f5ecd6;    /* neutral  */
    --bad:#b0503c; --bad-soft:#f7e7e2;    /* negative */
    --em:#2a5aa8; --em-soft:#e7eef8;
    --ncr:#7a5aa0; --ncr-soft:#f0eaf6;
    --serif:"Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,Georgia,serif;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;
  }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; padding:0; }}
  body {{ background:var(--paper); color:var(--ink); font-family:var(--sans); font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; }}
  .wrap {{ max-width:1060px; margin:0 auto; padding:56px 28px 72px; }}

  .eyebrow {{ font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:var(--faint); font-weight:600; margin:0 0 10px; }}
  .masthead {{ border-top:3px solid var(--ink); padding-top:20px; }}
  .masthead .rule {{ border-bottom:1px solid var(--line-strong); }}
  h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(30px,4.6vw,50px); line-height:1.05; letter-spacing:-.015em; margin:0 0 12px; max-width:24ch; }}
  .lede {{ font-size:17px; color:var(--muted); max-width:72ch; margin:0 0 6px; }}
  .meta-line {{ font-family:var(--mono); font-size:12px; color:var(--faint); }}

  .kpis {{ display:grid; grid-template-columns:1.4fr 1fr 1fr 1fr; border-top:1px solid var(--line-strong); margin:40px 0 6px; }}
  .kpi {{ padding:18px 22px 18px 0; border-bottom:1px solid var(--line); }}
  .kpi + .kpi {{ border-left:1px solid var(--line); padding-left:24px; }}
  .kpi .num {{ font-family:var(--serif); font-weight:600; font-size:38px; line-height:1; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }}
  .kpi .num .unit {{ font-size:18px; color:var(--muted); }}
  .kpi .lab {{ font-size:12px; color:var(--faint); text-transform:uppercase; letter-spacing:.08em; margin-top:8px; }}
  #kpiPos .num {{ color:var(--ok); }} #kpiNeu .num {{ color:var(--neu); }} #kpiNeg .num {{ color:var(--bad); }}

  section {{ margin-top:64px; }}
  .kicker {{ font-size:12px; font-weight:600; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); margin:0 0 6px; }}
  h2 {{ font-family:var(--serif); font-weight:600; font-size:28px; letter-spacing:-.01em; margin:0 0 8px; }}
  .sub {{ color:var(--muted); max-width:70ch; margin:0 0 26px; }}

  .panel {{ display:grid; grid-template-columns:240px 1fr; gap:52px; background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:30px 34px; align-items:center; }}
  .donut-wrap {{ text-align:center; }}
  .donut {{ position:relative; width:200px; height:200px; margin:0 auto 6px; }}
  .donut svg {{ transform:rotate(-90deg); }}
  .donut .center {{ position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
  .donut .center .big {{ font-family:var(--serif); font-size:44px; font-weight:600; line-height:1; }}
  .donut .center .cap {{ font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:var(--faint); margin-top:4px; }}
  .legend {{ display:flex; gap:18px; justify-content:center; font-size:12px; color:var(--muted); margin-top:10px; }}
  .legend span {{ display:inline-flex; align-items:center; gap:7px; }}
  .sw {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}

  .class-bars {{ display:grid; gap:24px; }}
  .cbar .row {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }}
  .cbar .clab {{ font-weight:600; font-size:14px; display:flex; align-items:center; gap:8px; }}
  .cbar .cval {{ font-family:var(--mono); font-size:12px; color:var(--muted); }}
  .dot {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}
  .dot.positive {{ background:var(--ok); }} .dot.neutral {{ background:var(--neu); }} .dot.negative {{ background:var(--bad); }}
  .track {{ display:flex; height:34px; border-radius:8px; overflow:hidden; background:var(--line); }}
  .track .pseg {{ height:100%; }}
  .pseg.seg-ok {{ background:var(--ok); }} .pseg.seg-neu {{ background:var(--neu); }} .pseg.seg-bad {{ background:var(--bad); }}
  .track-note {{ margin-top:8px; font-size:12px; color:var(--faint); }}

  .miss-list {{ display:grid; gap:16px; }}
  .miss {{ position:relative; background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:20px 22px 20px 24px; overflow:hidden; }}
  .miss-rail {{ position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--line-strong); }}
  .miss-meta {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; font-size:13px; }}
  .stars {{ color:#c9952f; letter-spacing:2px; font-size:14px; }}
  .chip {{ font-family:var(--mono); font-size:11px; font-weight:600; letter-spacing:.04em; padding:3px 9px; border-radius:999px; }}
  .chip.ok {{ background:var(--ok-soft); color:var(--ok); }}
  .chip.neu {{ background:var(--neu-soft); color:var(--neu); }}
  .chip.bad {{ background:var(--bad-soft); color:var(--bad); }}
  .chip.nl {{ background:var(--em-soft); color:var(--em); }}
  .chip.nr {{ background:var(--ncr-soft); color:var(--ncr); }}
  .vs {{ color:var(--faint); font-size:12px; }}
  .arrow {{ color:var(--faint); }}
  .miss-title {{ font-family:var(--serif); font-size:19px; font-weight:600; margin:12px 0 4px; }}
  .miss-text {{ margin:0 0 10px; }}
  .miss-why {{ font-size:13.5px; color:var(--muted); border-top:1px solid var(--line); padding-top:10px; margin:0; }}
  .miss-why b {{ color:var(--ink); }}

  .agree-band {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:24px 30px; margin-bottom:22px; display:flex; flex-wrap:wrap; gap:18px; align-items:baseline; }}
  .agree-num {{ font-family:var(--serif); font-weight:600; font-size:40px; line-height:1; color:var(--em); }}
  .agree-txt .ag {{ font-size:16px; font-weight:600; }}
  .agree-txt .an {{ color:var(--muted); font-size:13.5px; margin-top:4px; max-width:76ch; }}
  .dist-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .dist-panel {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:20px 24px; }}
  .dist-panel h4 {{ margin:0 0 4px; font-family:var(--serif); font-size:17px; font-weight:600; }}
  .dist-panel .cap {{ font-size:12px; color:var(--faint); margin-bottom:14px; }}
  .drow {{ display:grid; grid-template-columns:120px 1fr 34px; gap:10px; align-items:center; padding:3px 0; font-size:13px; }}
  .dbar {{ height:14px; border-radius:4px; background:var(--line); overflow:hidden; }}
  .dbar > div {{ height:100%; }}
  .dp-model .dbar > div {{ background:var(--em); }}
  .dp-nrc .dbar > div {{ background:var(--ncr); }}
  .dcnt {{ font-family:var(--mono); font-size:12px; color:var(--muted); text-align:right; }}

  .diff-list {{ display:grid; gap:14px; }}
  .diff {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 20px; }}
  .diff-top {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px; }}
  .diff-title {{ font-family:var(--serif); font-weight:600; font-size:16px; }}
  .diff-sent {{ margin-left:auto; font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--faint); }}
  .diff-mid {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; }}
  .emo-box {{ display:flex; align-items:center; gap:9px; }}
  .emo-lab {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--faint); }}
  .diff-why {{ margin:12px 0 0; font-size:13.5px; color:var(--muted); border-top:1px solid var(--line); padding-top:10px; }}
  .diff-why b {{ color:var(--ink); }}

  .controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  .filters {{ display:grid; gap:10px; margin-bottom:16px; }}
  .frow {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; border-bottom:1px solid var(--line); padding-bottom:10px; }}
  .frow:last-child {{ border-bottom:none; padding-bottom:0; }}
  .flabel {{ font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:var(--faint); font-weight:600; min-width:74px; }}
  .countbar {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:12px 18px; margin-bottom:16px; }}
  .bigcount {{ font-family:var(--serif); font-weight:600; font-size:22px; letter-spacing:-.01em; }}
  .bigcount .em {{ color:var(--ok); }}
  .countsub {{ font-size:13px; color:var(--muted); }}
  .countsub .m2 {{ color:var(--bad); font-weight:600; }}
  .countsub .ok2 {{ color:var(--ok); font-weight:600; }}
  .countsub .n3 {{ color:var(--neu); font-weight:600; }}
  .seg-btn {{ appearance:none; border:1px solid var(--line-strong); background:var(--surface); color:var(--muted); font-family:var(--sans); font-size:13px; font-weight:600; padding:8px 14px; border-radius:999px; cursor:pointer; transition:background .15s,color .15s,border-color .15s; }}
  .seg-btn:hover {{ border-color:var(--ink); color:var(--ink); }}
  .seg-btn.active {{ background:var(--ink); color:#fff; border-color:var(--ink); }}

  .table-wrap {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; overflow:auto; max-height:640px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  thead th {{ position:sticky; top:0; background:#fbfaf7; text-align:left; z-index:2; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--faint); padding:11px 12px; border-bottom:1px solid var(--line-strong); font-weight:600; }}
  tbody td {{ padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
  tbody tr:last-child td {{ border-bottom:none; }}
  td.idx {{ font-family:var(--mono); color:var(--faint); font-size:12px; width:40px; }}
  td.stars {{ white-space:nowrap; font-size:12px; }}
  .pill {{ font-family:var(--mono); font-size:11px; font-weight:600; padding:2px 8px; border-radius:999px; white-space:nowrap; }}
  .pill.ok {{ background:var(--ok-soft); color:var(--ok); }}
  .pill.neu {{ background:var(--neu-soft); color:var(--neu); }}
  .pill.bad {{ background:var(--bad-soft); color:var(--bad); }}
  .pill.nl {{ background:var(--em-soft); color:var(--em); }}
  .pill.nr {{ background:var(--ncr-soft); color:var(--ncr); }}
  td.tt {{ max-width:330px; }}
  td.tt .t {{ font-weight:600; }}
  td.tt .x {{ color:var(--muted); font-size:12px; }}
  tr.missrow {{ background:var(--paper); }}
  tr.agX {{ background:#f3efe6; }}
  .legend2 {{ display:flex; gap:16px; font-size:12px; color:var(--muted); margin-bottom:12px; }} .legend2 span {{ display:inline-flex; align-items:center; gap:6px; }} .legend2 i {{ width:11px; height:11px; border-radius:3px; }}

  /* ---- descriptive layer ---- */
  .glance-callout {{ background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--bad); border-radius:10px; padding:14px 18px; font-size:14px; color:var(--ink); margin-bottom:22px; }}
  .glance-callout b {{ color:var(--bad); }}
  .ov-grid {{ display:grid; grid-template-columns:1fr 1.2fr 1fr; gap:16px; }}
  .ov-card {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:20px 22px; }}
  .ov-card h4 {{ margin:0 0 4px; font-family:var(--serif); font-size:17px; font-weight:600; }}
  .ov-card .cap {{ font-size:12px; color:var(--faint); margin:0 0 16px; }}

  .stars-chart {{ display:flex; align-items:flex-end; gap:10px; }}
  .vcol {{ flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; }}
  .vcol .vval {{ font-family:var(--mono); font-size:12px; color:var(--muted); margin-bottom:4px; }}
  .vcol .vbar {{ width:26px; height:110px; background:var(--line); border-radius:4px; overflow:hidden; position:relative; }}
  .vcol .vfill {{ position:absolute; left:0; right:0; bottom:0; }}
  .vcol .vlab {{ font-size:11px; color:var(--faint); margin-top:6px; }}

  .rp-chart {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
  .rp-group .rp-head {{ font-size:12px; font-weight:600; display:flex; align-items:center; gap:6px; margin-bottom:8px; }}
  .rp-group .rp-nums {{ font-family:var(--mono); font-size:11px; color:var(--faint); font-weight:400; margin-left:auto; }}
  .rp-cols {{ display:flex; gap:12px; justify-content:center; align-items:flex-end; height:120px; }}
  .rp-col {{ display:flex; flex-direction:column; align-items:center; justify-content:flex-end; }}
  .rp-col .vvar {{ font-family:var(--mono); font-size:12px; color:var(--muted); margin-bottom:4px; }}
  .rp-col .vbar {{ width:22px; height:90px; background:var(--line); border-radius:4px; overflow:hidden; position:relative; }}
  .rp-col .vfill {{ position:absolute; left:0; right:0; bottom:0; background:var(--line-strong); }}
  .rp-col .vlab {{ font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--faint); margin-top:5px; }}
  .rp-legend {{ display:flex; gap:16px; justify-content:center; margin-top:14px; font-size:12px; color:var(--muted); }}
  .rp-legend span {{ display:inline-flex; align-items:center; gap:6px; }}

  .acc-chart {{ display:grid; gap:14px; }}
  .abar-lab {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:5px; font-size:13px; }}
  .abar-name {{ font-weight:600; display:inline-flex; align-items:center; gap:7px; }}
  .abar-v {{ font-family:var(--mono); font-size:12px; color:var(--muted); }}
  .abar {{ height:16px; background:var(--line); border-radius:4px; overflow:hidden; }}
  .abar-fill {{ height:100%; }}

  @media (max-width:900px){{\n    .ov-grid {{ grid-template-columns:1fr; }}\n  }}

  footer {{ margin-top:60px; padding-top:22px; border-top:1px solid var(--line-strong); display:grid; gap:6px; color:var(--faint); font-size:12.5px; }}
  footer .h {{ text-transform:uppercase; letter-spacing:.1em; font-size:11px; color:var(--muted); font-weight:600; }}
  footer p {{ margin:0; max-width:86ch; }}
  footer code {{ font-family:var(--mono); font-size:11.5px; background:var(--surface); padding:1px 5px; border-radius:4px; }}

  @media (max-width:840px) {{
    .kpis {{ grid-template-columns:1fr 1fr; }}
    .kpi + .kpi {{ border-left:none; }}
    .kpi:nth-child(3),.kpi:nth-child(4) {{ border-top:1px solid var(--line); }}
    .panel {{ grid-template-columns:1fr; gap:34px; }}
    .dist-grid {{ grid-template-columns:1fr; }}
  }}
  @media (prefers-reduced-motion:no-preference) {{ .masthead h1, .kpis .kpi, section {{ animation:fade .5s ease both; }} }}
  @keyframes fade {{ from{{opacity:0; transform:translateY(6px);}} to{{opacity:1; transform:none;}} }}
</style>
</head>
<body>
<div class="wrap">

  <header class="masthead">
    <p class="eyebrow">Amazon Gift Card Reviews &middot; Three-Class Sentiment &amp; Emotion</p>
    <h1>{agree} of {n} balanced reviews matched the star rating.</h1>
    <p class="lede">
      Sentiment is now <b>three classes</b>: positive (4&ndash;5&#9733;), neutral (3&#9733;), negative
      (1&ndash;2&#9733;). A model reads each review&rsquo;s <em>title and text only</em> and picks one class,
      compared against the rating (used only as the correct answer, never shown to the model). Each review also
      gets a <b>primary emotion</b> from the model and from the NRC word-list. To stop the rare classes being
      swamped, the sample is balanced &mdash; {per_cls} of each rating class &mdash; and picked with a fixed random seed.
    </p>
    <p class="meta-line">{SRC_NOTE} &middot; offline dashboard &middot; rubric: 4&ndash;5 = positive, 3 = neutral, 1&ndash;2 = negative</p>
    <div class="rule"></div>
  </header>

  <div class="kpis">
    <div class="kpi lead"><div class="num">{agree}<span class="unit">/{n} &middot; {pct}%</span></div><div class="lab">Overall match (3-class)</div></div>
    <div class="kpi" id="kpiPos"><div class="num">{by_cls["positive"]["correct"]}<span class="unit">/{by_cls["positive"]["n"]} &middot; {by_cls["positive"]["acc"]}%</span></div><div class="lab">Positive matched</div></div>
    <div class="kpi" id="kpiNeu"><div class="num">{by_cls["neutral"]["correct"]}<span class="unit">/{by_cls["neutral"]["n"]} &middot; {by_cls["neutral"]["acc"]}%</span></div><div class="lab">Neutral matched</div></div>
    <div class="kpi" id="kpiNeg"><div class="num">{by_cls["negative"]["correct"]}<span class="unit">/{by_cls["negative"]["n"]} &middot; {by_cls["negative"]["acc"]}%</span></div><div class="lab">Negative matched</div></div>
  </div>

  {overview_html}

  <section id="rightwrong">
    <p class="kicker">Sentiment &middot; right vs. wrong</p>
    <h2>{agree} matched, {miss} missed</h2>
    <p class="sub">
      Each bar shows one rating class and how the model labelled its reviews: green = called positive,
      amber = neutral, red = negative. The coloured segment that matches the class is a hit.
      Notice the weak spot: over half the truly-neutral (3-star) reviews were read as negative.
    </p>
    <div class="panel">
      <div class="donut-wrap">
        <div class="donut" id="donut"></div>
        <div class="legend">
          <span><i class="sw" style="background:var(--ok)"></i>matched</span>
          <span><i class="sw" style="background:var(--bad)"></i>missed</span>
        </div>
      </div>
      <div class="class-bars">{class_bars_html}</div>
    </div>
  </section>

  <section id="smisses">
    <p class="kicker">Sentiment &middot; example misses</p>
    <h2>A look at the {miss} reviews the model missed</h2>
    <p class="sub">A representative handful (three per class), so you can read them yourself. The full list with every review is in the evidence table below.</p>
    <div class="miss-list">{miss_cards_html}</div>
  </section>

  <section id="emotion">
    <p class="kicker">Primary emotion &middot; two independent methods</p>
    <h2>How often the model and the word-list agree</h2>
    <p class="sub">
      <b>The model</b> reads the review and names one emotion. <b>The NRC word-list</b> is a plain lookup table
      (NRC Emotion Lexicon) that counts emotion words in the review text and takes the highest category &mdash;
      no model call. Both answer: what is this review&rsquo;s single primary emotion?
    </p>
    <div class="agree-band">
      <div class="agree-num">{e_pct}%</div>
      <div class="agree-txt">
        <div class="ag">The two methods chose the same primary emotion {e_agree} of {n} times.</div>
        <div class="an">Agreement is low because the methods differ in kind. The model reads meaning; the
          word-list can be tipped by common words it scores as &ldquo;anticipation&rdquo; or &ldquo;joy&rdquo;. In
          {e_none} reviews the text held no NRC emotion words, so the word-list could not choose. Their category
          choices are compared below.</div>
      </div>
    </div>
    <div class="dist-grid">
      <div class="dist-panel dp-model"><h4>What the model picked</h4><div class="cap">Primary emotion chosen by the LLM &mdash; one per review.</div><div id="distModel"></div></div>
      <div class="dist-panel dp-nrc"><h4>What the NRC word-list picked</h4><div class="cap">Highest word-tally category per review &mdash; no model calls.</div><div id="distNrc"></div></div>
    </div>
  </section>

  <section id="emodiffs">
    <p class="kicker">Primary emotion &middot; where they disagree</p>
    <h2><span id="diffCount"></span> reviews where the model and the word-list disagreed</h2>
    <p class="sub">A sample of disagreements with the words the word-list counted. Use the &ldquo;Emotion match&rdquo; filter below to page through every case.</p>
    <div class="diff-list">{diff_cards_html}</div>
  </section>

  <section id="evidence">
    <p class="kicker">Evidence</p>
    <h2>Every review, one row</h2>
    <p class="sub">All {n} balanced reviews with three-class sentiment and both emotion labels. Use the filters to explore; the count bar updates live.</p>
    <div class="legend2">
      <span><i style="background:var(--ok)"></i>positive</span>
      <span><i style="background:var(--neu)"></i>neutral</span>
      <span><i style="background:var(--bad)"></i>negative</span>
    </div>
    <div class="filters">
      <div class="frow"><span class="flabel">Result</span>
        <div class="controls" data-group="result">
          <button class="seg-btn active" data-val="all">All</button>
          <button class="seg-btn" data-val="ok">Matched</button>
          <button class="seg-btn" data-val="miss">Mismatched</button>
        </div></div>
      <div class="frow"><span class="flabel">By class</span>
        <div class="controls" data-group="cls">
          <button class="seg-btn active" data-val="all">All</button>
          <button class="seg-btn" data-val="positive">Positive</button>
          <button class="seg-btn" data-val="neutral">Neutral</button>
          <button class="seg-btn" data-val="negative">Negative</button>
        </div></div>
      <div class="frow"><span class="flabel">Stars</span>
        <div class="controls" data-group="stars">
          <button class="seg-btn active" data-val="all">All stars</button>
          <button class="seg-btn" data-val="1">1&#9733;</button>
          <button class="seg-btn" data-val="2">2&#9733;</button>
          <button class="seg-btn" data-val="3">3&#9733;</button>
          <button class="seg-btn" data-val="4">4&#9733;</button>
          <button class="seg-btn" data-val="5">5&#9733;</button>
        </div></div>
      <div class="frow"><span class="flabel">Emotion match</span>
        <div class="controls" data-group="emo">
          <button class="seg-btn active" data-val="all">All</button>
          <button class="seg-btn" data-val="agree">Methods agree</button>
          <button class="seg-btn" data-val="differ">Methods differ</button>
        </div></div>
    </div>
    <div class="countbar">
      <span class="bigcount" id="bigcount"></span>
      <span class="countsub" id="countsub"></span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>#</th><th>Stars</th><th>Rating says</th><th>Model read</th>
          <th>Model emotion</th><th>NRC emotion</th><th>Emotion match</th><th>Title &amp; text</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </section>

  <footer>
    <span class="h">Method &amp; caveats</span>
    <p>Sample: {n} balanced reviews ({per_cls} per rating class) drawn from the whole file with a fixed random seed {seed_val}; the same set is produced every run.</p>
    <p>Sentiment: title + text only, never the rating. Rubric: 4&ndash;5 = positive, 3 = neutral, 1&ndash;2 = negative. Rating used only as the correct answer for scoring.</p>
    <p>Primary emotion (model): one LLM call per review, Qwen3.6-35B, chain-of-thought disabled. Primary emotion (NRC): NRC-EmoLex v0.92 word-level; highest word-tally per category wins; ties broken by a fixed order; no model calls.</p>
    <p>Neutral is the hardest class: 3-star reviews are often written with complaint-heavy text, so the model frequently reads them as negative.</p>
    <p>Offline, no server or network required &mdash; everything is embedded in this single file.</p>
  </footer>

</div>

<script>
const ROWS = {ROWS_JSON};
const ORDERC = {ORDER_JSON};
const MODEL_DIST = {json.dumps(model_series)};
const NRC_DIST   = {json.dumps(nrc_series)};
const E_AGREE = {e_agree}, E_NONE = {e_none}, N = {n};
const CLS_ORDER = ["positive","neutral","negative"];
const CLS_KEY = {{"positive":"ok","neutral":"neu","negative":"bad"}};
const PILL = {{"anger":"bad","anticipation":"nl","disgust":"bad","fear":"bad","joy":"ok","sadness":"bad","surprise":"nl","trust":"ok","none":"nr"}};
const EMO8 = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"];

(function(){{
  const svgNS="http://www.w3.org/2000/svg";
  const host=document.getElementById("donut");
  const R=86,C=2*Math.PI*R, svg=document.createElementNS(svgNS,"svg");
  svg.setAttribute("width",200);svg.setAttribute("height",200);
  const mk=(color,frac,off)=>{{ const c=document.createElementNS(svgNS,"circle");
    c.setAttribute("cx",100);c.setAttribute("cy",100);c.setAttribute("r",R);c.setAttribute("fill","none");
    c.setAttribute("stroke",color);c.setAttribute("stroke-width",26);
    c.setAttribute("stroke-dasharray",(frac*C).toFixed(1)+" "+(C).toFixed(1));
    c.setAttribute("stroke-dashoffset",-off*C);c.setAttribute("stroke-linecap","round");return c; }};
  const ok={agree}/N;
  const okC=getComputedStyle(document.documentElement).getPropertyValue("--ok")||"#1f7a5c";
  const badC=getComputedStyle(document.documentElement).getPropertyValue("--bad")||"#b0503c";
  svg.appendChild(mk(okC,ok,0)); svg.appendChild(mk(badC,1-ok,ok));
  host.appendChild(svg);
  const c=document.createElement("div");c.className="center";
  c.innerHTML='<div class="big">'+({pct})+'%</div><div class="cap">matched</div>';
  host.appendChild(c);
}})();

function distChart(id,series){{ const host=document.getElementById(id);
  const m=Math.max.apply(null,series)||1;
  ORDERC.forEach((k,i)=>{{ if(series[i]===0)return;
    const d=document.createElement("div");d.className="drow";
    d.innerHTML='<span>'+k+'</span><div class="dbar"><div style="width:'+Math.max(100*series[i]/m,3)+'%"></div></div><span class="dcnt">'+series[i]+'</span>';
    host.appendChild(d); }}); }}
distChart("distModel",MODEL_DIST); distChart("distNrc",NRC_DIST);
document.getElementById("diffCount").textContent = (N - E_AGREE);

const tbody=document.getElementById("tbody"); const bigcount=document.getElementById("bigcount"); const countsub=document.getElementById("countsub");
const state={{"result":"all","cls":"all","stars":"all","emo":"all"}};
const stars2=s=>"\\u2605".repeat(Math.round(s));
function esc(s){{ return (s||"").replace(/[&<>"]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c])); }}
function sentPill(v){{ return '<span class="pill '+CLS_KEY[v]+'">'+v+'</span>'; }}
function emoPill(v){{ const k=PILL[v]||"bad"; return '<span class="pill '+k+'">'+v+'</span>'; }}
function emAg(r){{ return r["model_emotion"] in PILL && r["model_emotion"] !== "none" && r["model_emotion"]===r["nrc_emotion"]; }}

function matches(r){{
  const ok=r.correct;
  if(state["result"]==="ok"&&!ok)return false;
  if(state["result"]==="miss"&&ok)return false;
  if(state["cls"]!=="all"&&r["cls"]!==state["cls"])return false;
  if(state["stars"]!=="all"&&Math.round(r["rating"])!==Number(state["stars"]))return false;
  if(state["emo"]==="agree"&&!emAg(r))return false;
  if(state["emo"]==="differ"&&emAg(r))return false;
  return true;
}}
function apply(){{
  const show=ROWS.filter(matches);
  const ok=show.filter(r=>r.correct).length;
  const ea=show.filter(emAg).length;
  const byc={{positive:0,neutral:0,negative:0}};
  show.forEach(r=>byc[r["cls"]]++);
  tbody.innerHTML="";
  show.forEach(r=>{{
    const tr=document.createElement("tr");
    if(!r.correct)tr.className="missrow"; else if(!emAg(r))tr.className="agX";
    tr.innerHTML='<td class="idx">'+r.id+'</td>'+
      '<td class="stars">'+stars2(r.rating)+'</td>'+
      '<td>'+sentPill(r.cls)+'</td><td>'+sentPill(r.sentiment)+'</td>'+
      '<td>'+emoPill(r.model_emotion)+'</td><td>'+emoPill(r.nrc_emotion)+'</td>'+
      '<td>'+(emAg(r)?'<span class="pill ok">agree</span>':'<span class="pill bad">differ</span>')+'</td>'+
      '<td class="tt"><span class="t">'+esc(r.title)+'</span><br><span class="x">'+esc(r.text)+'</span></td>';
    tbody.appendChild(tr);
  }});
  bigcount.innerHTML='<span class="em">'+show.length+'</span> of '+ROWS.length+' reviews';
  countsub.innerHTML=(show.length===ROWS.length?'All reviews':'Showing '+show.length+' of '+ROWS.length)+
    ' &middot; <span class="ok2">'+ok+' matched</span> &middot; <span class="m2">'+(show.length-ok)+' mismatched</span>'+
    ' &middot; <span class="n3">pos '+byc["positive"]+'</span> / <span class="n3">neu '+byc["neutral"]+'</span> / <span class="m2">neg '+byc["negative"]+'</span>'+
    ' &middot; emotion <span class="ok2">'+ea+' agree</span> / <span class="m2">'+(show.length-ea)+' differ</span>';
}}
document.querySelectorAll(".controls").forEach(g=>{{
  const gn=g.dataset.group;
  g.querySelectorAll(".seg-btn").forEach(b=>b.addEventListener("click",()=>{{
    g.querySelectorAll(".seg-btn").forEach(x=>x.classList.remove("active"));
    b.classList.add("active"); state[gn]=b.dataset.val; apply();
  }}));
}});
console.assert(ROWS.filter(r=>r.correct).length==={agree},"agree mismatch");
console.assert(ROWS.length==={n},"row mismatch");
apply();
</script>
</body>
</html>
"""

HTML = HTML.replace("\\u2605", "\u2605")  # JS: escape -> literal star

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
import os
print("wrote", OUT, os.path.getsize(OUT), "bytes")
