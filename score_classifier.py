#!/usr/bin/env python3
"""
Score the finished classifier on the FIRST 100 reviews of Gift_Cards.jsonl.gz.
The rating is used ONLY afterward as ground truth (never fed to the model):
    4-5 stars  ->  POSITIVE  (correct answer)
    1-3 stars  ->  NEGATIVE  (correct answer)
Prediction comes from title+text only, via classifier.classify().
"""
import csv, gzip, json, time
from concurrent.futures import ThreadPoolExecutor
from classifier import classify

INPUT  = "Gift_Cards.jsonl.gz"
N      = 100
OUTJ   = "scoring_results.json"
OUTCSV = "scoring_results.csv"

def load_first(n):
    recs = []
    with gzip.open(INPUT, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
            if len(recs) >= n:
                break
    return recs

def target(rec):
    return "positive" if rec["rating"] >= 4 else "negative"

def main():
    recs = load_first(N)
    print("loaded %d reviews" % len(recs), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        preds = list(ex.map(lambda r: classify(r.get("title", ""), r.get("text", "")), recs))
    print("classified all %d in %.1fs" % (len(recs), time.time() - t0), flush=True)

    rows = []
    for i, (rec, pred) in enumerate(zip(recs, preds)):
        tg = target(rec)
        rows.append({
            "idx": i, "rating": rec.get("rating"), "target": tg, "pred": pred,
            "correct": (pred == tg), "title": rec.get("title") or "",
            "text": rec.get("text") or "",
        })

    n = len(rows)
    n_agree = sum(1 for r in rows if r["correct"])
    wrong = [r for r in rows if not r["correct"]]

    by_tgt = {}
    for tg in ("positive", "negative"):
        sub = [r for r in rows if r["target"] == tg]
        ok = sum(1 for r in sub if r["correct"])
        by_tgt[tg] = {
            "n": len(sub),
            "correct": ok,
            "class_accuracy": (ok / len(sub)) if sub else None,
        }
        if ok != len(sub):
            by_tgt[tg]["wrong_indices"] = [r["idx"] for r in sub if not r["correct"]]

    out = {
        "rubric": "4-5 stars = positive ; 1-3 stars = negative (rating used only as ground truth, never in the model)",
        "n": n,
        "n_agree": n_agree,
        "overall_agreement": n_agree / n,
        "by_class": by_tgt,
        "wrong": [{k: r[k] for k in ("idx", "rating", "target", "pred", "title", "text")} for r in wrong],
        "rows": rows,
    }
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(OUTCSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "rating", "target", "pred", "correct", "title", "text"])
        for r in rows:
            w.writerow([r["idx"], r["rating"], r["target"], r["pred"], r["correct"],
                        r["title"], r["text"]])

    print("\n===== SCORING SUMMARY (first %d reviews) =====" % n)
    print("rubric: 4-5 stars = positive ; 1-3 stars = negative")
    print("agree with rating: %d/%d  = %.1f%%" % (n_agree, n, 100.0 * n_agree / n))
    for tg in ("positive", "negative"):
        b = by_tgt[tg]
        if b["n"]:
            print("class %s: %d examples, correct %d, class accuracy %.1f%%" % (
                tg, b["n"], b["correct"], 100.0 * b["class_accuracy"]))
        else:
            print("class %s: 0 examples in this sample" % tg)
    print("\n%d wrong:" % len(wrong))
    for r in wrong:
        print("  #%3d  rating=%.1f target=%s pred=%s | %s -- %s" % (
            r["idx"], r["rating"], r["target"], r["pred"], r["title"][:30], r["text"][:40]))

    print("\nsaved -> %s  and  %s" % (OUTJ, OUTCSV))

if __name__ == "__main__":
    main()
