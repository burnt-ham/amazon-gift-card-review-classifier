#!/usr/bin/env python3
"""
Clean classifier for Amazon Gift Card reviews.
Input : a review's `title` and `text`  ->  Output: `positive` or `negative`.

Only title + text are ever sent to the model. The star rating is NEVER used here.
Output is a single clean word so it can be read back programmatically.

Usage as a library:
    from classifier import classify
    label = classify("Great gift", "The card loaded instantly and worked perfectly.")
    # label -> "positive"

Usage as a demo:
    python classifier.py        # labels a small sample of reviews from the dataset
"""
import json
import os
import urllib.request

BASE    = "http://dobolyi.com:9001/v1/chat/completions"
MODEL   = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

# The API key is read from the LLM_API_KEY environment variable so that no
# credential is committed to this (public) repository. Set it before running,
# e.g.  export LLM_API_KEY=<key>   (value supplied in the assignment)
API_KEY = os.environ.get("LLM_API_KEY", "")


def _auth():
    if not API_KEY:
        raise RuntimeError(
            "No LLM_API_KEY set. Export the assignment's endpoint key first, e.g. "
            "export LLM_API_KEY=...  on bash, or set $env:LLM_API_KEY=... in PowerShell."
        )
    return "Bearer " + API_KEY

# The NRC emotion categories the model chooses one primary emotion from.
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

_SYSTEM = (
    "You classify Amazon product reviews as positive, neutral, or negative based ONLY on the "
    "review's title and its text. You are never given a star rating, so do not assume one. "
    "Decide by the overall expressed sentiment: when the title and the text conflict, give more "
    "weight to a substantive text body. Treat short, terse, informal, or sarcastic reviews the "
    "same way and judge them by the words used. Use 'neutral' when the review expresses no clear "
    "positive or negative feeling (matter-of-fact, mixed, or unemotional). "
    "Reply with exactly one word: positive or neutral or negative. No punctuation, reasoning, or extra text."
)

_EMOTION_SYSTEM = (
    _SYSTEM +
    " Then, on the same line, state the ONE primary emotion the review's author most strongly "
    "expresses, choosing only from these eight: anger, anticipation, disgust, fear, joy, sadness, "
    "surprise, trust. "
    "Output format: exactly two words separated by a single space: <positive|neutral|negative> <emotion>. "
    "For example: 'positive joy' or 'negative anger' or 'neutral anticipation'. "
    "No punctuation, reasoning, or extra text."
)

def classify(title, text, timeout=60):
    """Return 'positive' or 'negative' for the given review title+text."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "Review title: %s\nReview text: %s" % (title, text)},
        ],
        "max_tokens": 10,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},  # force direct answer, no chain-of-thought
    }
    req = urllib.request.Request(
        BASE,
        data=json.dumps(payload).encode(),
        headers={"Authorization": _auth(), "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = (json.loads(r.read().decode())["choices"][0]["message"].get("content") or "").strip().lower()
    if content.startswith("positive"):
        return "positive"
    if content.startswith("neutral"):
        return "neutral"
    if content.startswith("negative"):
        return "negative"
    raise ValueError("Unparseable model output: %r" % content[:40])


def classify_sentiment_emotion(title, text, timeout=60):
    """Return {'sentiment', 'emotion', 'raw'} for a review, in one model call.

    sentiment is 'positive'|'neutral'|'negative'; emotion is one of the 8 NRC categories
    (or 'other:<word>' if the model produced an unrecognized label, for transparency);
    'raw' is the model's verbatim text for inspection.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _EMOTION_SYSTEM},
            {"role": "user", "content": "Review title: %s\nReview text: %s" % (title, text)},
        ],
        "max_tokens": 12,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE,
        data=json.dumps(payload).encode(),
        headers={"Authorization": _auth(), "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = (json.loads(r.read().decode())["choices"][0]["message"].get("content") or "").strip().lower()

    toks = raw.split()
    sentiment = toks[0] if toks and toks[0] in ("positive", "neutral", "negative") else "unparsed:" + raw[:24]
    emotion = toks[1] if len(toks) > 1 else "none"
    if emotion not in EMOTIONS:
        emotion = "other:" + emotion[:16]
    return {"sentiment": sentiment, "emotion": emotion, "raw": raw}


# --------------------------------------------------------------------------- demo
def _demo():
    import gzip
    with gzip.open("Gift_Cards.jsonl.gz", "rt", encoding="utf-8", errors="replace") as f:
        recs = [json.loads(line) for line in f if line.strip()][:8]

    print("%-8s %-20s %-60s" % ("pred", "title", "text"))
    print("-" * 96)
    for r in recs:
        label = classify(r.get("title", ""), r.get("text", ""))
        print("%-8s %-20.20s %-60.60s" % (label, r.get("title", ""), r.get("text", "")))

if __name__ == "__main__":
    _demo()
