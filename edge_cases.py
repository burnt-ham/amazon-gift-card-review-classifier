#!/usr/bin/env python3
"""
Demonstrates how the clean classifier handles tricky inputs: conflicting title/text,
and short / terse / sarcastic / angry reviews. Pure demonstration -- it just prints
the model's decision, with no correctness scoring.
"""
from classifier import classify

CASES = [
    ("conflict (title says good, text complains)", "Great gift!", "The code would not load and support was useless. Complete waste of money."),
    ("conflict (title says bad, text is happy)",  "Terrible", "Actually it worked perfectly and the balance loaded instantly. Very happy!"),
    ("terse angry",    "SCAM", "worst"),
    ("terse sarcasm",  "Loved it", "Says $50 but only $40 loaded. Truly a delight. Would pay full price for less again."),
    ("terse positive", "Great", "Perfect, thanks!"),
    ("angry short",    "Rubbish", "do not buy"),
    ("simple positive","Good card", "Loaded fine and works as expected."),
    ("mixed -> neg",   "OK I guess", "Loaded fast but then they froze the balance and demanded ID, awful experience."),
]

for cat, ti, tx in CASES:
    p = classify(ti, tx)
    print("%-38s %-9s | %s" % (cat, p, ti))
