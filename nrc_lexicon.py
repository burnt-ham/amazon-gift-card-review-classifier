#!/usr/bin/env python3
"""
NRC emotion word-list scorer.

Loads the official NRC Word-Emotion Association Lexicon (NRC-EmoLex, v0.92) and scores
review text against its 8 emotion categories. This is a purely local word-matching method:
NO model calls. The category with the highest word-tally is the word-list prediction.

Categories (canonical NRC set): anger, anticipation, disgust, fear, joy, sadness,
surprise, trust. (Note: the task brief printed "dishust" — the correct NRC label is "disgust".)

Tie-breaking: when two categories tie for highest tally, the one appearing first in the
canonical order above wins, so results are deterministic.
"""
import os, re

# Canonical NRC emotion categories, in a fixed order (serves as tie-break order).
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

DEFAULT_LEXICON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nrc", "NRC-Emotion-Lexicon", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

_TOKEN_RE = re.compile(r"[a-z']+")


def load_lexicon(path=DEFAULT_LEXICON):
    """Return {word: set_of_emotions} built from the NRC-EmoLex word-level file.

    Each line is `word<TAB>emotion<TAB>1|0`; we keep only 1-valued associations and only
    the 8 emotion columns (we ignore the positive/negative valence rows).
    """
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, val = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if emotion not in EMOTIONS or val != "1":
                continue
            lex.setdefault(word, set()).add(emotion)
    return lex


def score_text(text, lexicon):
    """Return (tallies_dict, matched_words) for one review's text.

    tallies_dict: {emotion: count_of_matching_words}
    matched_words: dict {emotion: [words that contributed]}
    """
    tallies = {e: 0 for e in EMOTIONS}
    matched = {e: [] for e in EMOTIONS}
    if not text:
        return tallies, matched
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("'")
        if tok in lexicon:
            for emo in lexicon[tok]:
                tallies[emo] += 1
                matched[emo].append(tok)
    return tallies, matched


def primary_emotion(text, lexicon=None):
    """Return the highest-scoring NRC emotion for the text, or "none" if no word matched.

    Optional extra: also returns the full tallies dict (mapped by caller via score_text).
    """
    lex = lexicon if lexicon is not None else load_lexicon()
    tallies, _ = score_text(text, lex)
    best = max(EMOTIONS, key=lambda e: tallies[e])
    if tallies[best] == 0:
        return "none", tallies
    return best, tallies


def distribution(corpus, lexicon=None):
    """Return {emotion: count} of NRC primary emotions across a list of texts."""
    lex = lexicon if lexicon is not None else load_lexicon()
    from collections import Counter
    c = Counter()
    for t in corpus:
        emo, _ = primary_emotion(t, lex)
        c[emo] += 1
    return dict(c)


if __name__ == "__main__":
    lex = load_lexicon()
    print("loaded NRC lexicon: %d words" % len(lex))
    for s in ["I love this gift card, it is wonderful and delightful",
              "This is terrible, I am so angry and frustrated",
              "Surprise! that was unexpected and astonishing",
              "It's completely neutral, tables and chairs and levels",
              "abandon abandon fear dread terror"]:
        emo, tall = primary_emotion(s, lex)
        print("%-60s -> %-13s %s" % (s[:58], emo, tall))
