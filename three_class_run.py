#!/usr/bin/env python3
"""
Three-class sentiment + emotion on a BALANCED sample of the Gift Card reviews.

Sample: ~50 reviews from EACH rating class, pulled from the WHOLE file with a fixed
random seed so the same set appears every run. Classes (by rating):
     4-5 = positive, 3 = neutral, 1-2 = negative.
For each sampled review we compute, in one model call: three-class sentiment + model emotion,
and (locally, no calls) the NRC word-list emotion. Raw model output is saved too.

Output: three_class_results.json
"""
import gzip, json, random, time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from classifier import classify_sentiment_emotion, EMOTIONS
from nrc_lexicon import load_lexicon, score_text

INPUT = "Gift_Cards.jsonl.gz"
OUT   = "three_class_results.json"
SEED  = 2026
PER   = 50

def rating_class(rv):
    rv = int(round(rv))
    if rv >= 4:
        return "positive"
    if rv == 3:
        return "neutral"
    return "negative"

def load_all(path):
    recs = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs

def main():
    print("loading all records...", flush=True)
    recs = load_all(INPUT)
    print("total records:", len(recs), flush=True)

    buckets = {"positive": [], "neutral": [], "negative": []}
    for idx, r in enumerate(recs):
        rv = r.get("rating")
        if rv is None:
            continue
        buckets[rating_class(rv)].append((idx, r))

    # fixed seed -> same sample every time
    random.seed(SEED)
    sample = []
    class_counts = {}
    order = ["positive", "neutral", "negative"]
    for c in order:
        pool = buckets[c]
        k = min(PER, len(pool))
        picked = random.sample(pool, k)
        class_counts[c] = k
        sample.extend(picked)
    print("balanced sample:", class_counts, "total", len(sample), flush=True)

    random.seed(SEED)  # keep deterministic ordering for the saved file
    random.shuffle(sample)
    # sort into fixed order by class then original index so file order is stable
    sample.sort(key=lambda t: (order.index(rating_class(t[1]["rating"])), t[0]))
    for j, (_, r) in enumerate(sample):
        r["_id"] = j

    lexicon = load_lexicon()
    print("NRC lexicon ready", flush=True)

    def model_part(item):
        _, r = item
        return classify_sentiment_emotion(r.get("title", ""), r.get("text", ""))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        model_out = list(ex.map(model_part, sample))
    print("model classifications done in %.1fs" % (time.time() - t0), flush=True)

    rows = []
    for (idx, r), m in zip(sample, model_out):
        tallies, matched = score_text(r.get("text", ""), lexicon)
        best = max(EMOTIONS, key=lambda e: tallies[e])
        nrc_emotion = best if tallies[best] > 0 else "none"
        cls = rating_class(r["rating"])
        sent = m["sentiment"]
        rows.append({
            "id": r["_id"],
            "src_idx": idx,
            "title": r.get("title", ""), "text": r.get("text", ""),
            "rating": r.get("rating"), "cls": cls,
            "sentiment": sent,
            "raw": m["raw"],
            "correct": (sent == cls),
            "model_emotion": m["emotion"],
            "nrc_emotion": nrc_emotion,
            "nrc_tallies": tallies,
            "nrc_matched": matched,
        })

    n = len(rows)
    agree = sum(1 for r in rows if r["correct"])
    by_cls = {}
    for c in order:
        sub = [r for r in rows if r["cls"] == c]
        preds = Counter(r["sentiment"] for r in sub)
        by_cls[c] = {
            "n": len(sub),
            "correct": sum(1 for r in sub if r["correct"]),
            "acc": round(100.0 * sum(1 for r in sub if r["correct"]) / len(sub), 1),
            "pred_distribution": {k: preds.get(k, 0) for k in order},
            "wrong_prediction_counts": {k: preds.get(k, 0) for k in order if k != c},
        }

    # emotion agreement
    e_agree = [r for r in rows if r["model_emotion"] in EMOTIONS and r["model_emotion"] == r["nrc_emotion"]]
    nrc_none = [r for r in rows if r["nrc_emotion"] == "none"]
    model_dist = Counter(r["model_emotion"] for r in rows)
    nrc_dist = Counter(r["nrc_emotion"] for r in rows)

    result = {
        "sampling": {"seed": SEED, "per_class": PER, "classes": class_counts, "total": n,
                     "note": "balanced by rating class from the whole file, fixed seed"},
        "rubric": "4-5 = positive, 3 = neutral, 1-2 = negative (rating used only as ground truth, never in the prompt)",
        "sentiment_overall": {"total": n, "agree": agree, "agree_pct": round(100.0 * agree / n, 1),
                              "miss": n - agree},
        "class_accuracy": by_cls,
        "emotion_agreement": {"agree": len(e_agree), "agree_pct_all": round(100.0 * len(e_agree) / n, 1),
                              "nrc_none": len(nrc_none), "disagree": n - len(e_agree)},
        "model_emotion_distribution": {k: v for k, v in model_dist.items()},
        "nrc_emotion_distribution": {k: v for k, v in nrc_dist.items()},
        "rows": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== THREE-CLASS RESULTS (%d balanced reviews, seed %d) =====" % (n, SEED))
    for c in order:
        b = by_cls[c]
        print("%-8s n=%3d correct=%3d acc=%5.1f%%  preds=%s" % (
            c, b["n"], b["correct"], b["acc"], b["pred_distribution"]))
    print("overall match (3-class): %d/%d = %.1f%%  (misses %d)" % (agree, n, 100.0 * agree / n, n - agree))
    print("\nemotion: model/NRC agree %d/%d = %.1f%% (nrc none %d)" % (
        len(e_agree), n, 100.0 * len(e_agree) / n, len(nrc_none)))
    print("model emo dist:", dict(sorted(model_dist.items())))
    print("nrc   emo dist:", dict(sorted(nrc_dist.items())))
    unparsed = [r for r in rows if r["sentiment"].startswith("unparsed")]
    print("\nunparsed sentiment rows:", len(unparsed))
    print("saved -> %s" % OUT)


if __name__ == "__main__":
    main()
