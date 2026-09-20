"""Broader OpenJEV usage tour: predict, predict_hypotheses, rerank, grade, latents.

Run after test_jev.py passes:
    python3 test_examples.py
"""
import torch
from modeling_openjev import OpenJevCrossEncoder

MODEL_DIR = "."
SUBFOLDER = "qwen3.5-4b-nli-v2"


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Loading OpenJEV ({SUBFOLDER}) onto {device}...")
jev = OpenJevCrossEncoder(MODEL_DIR, subfolder=SUBFOLDER)

# 1. predict() — raw premise/hypothesis pairs -------------------------------
section("1. predict() — raw premise/hypothesis pairs")
pairs = [
    ("The AUV battery is at 18% and falling.", "The mission should return to the surface soon."),
    ("The AUV battery is at 18% and falling.", "The AUV has plenty of power for a 6-hour dive."),
]
for (p, h), s in zip(pairs, jev.predict(pairs)):
    print(f"P: {p}\nH: {h}")
    print(f"  contradiction={s[0]:.3f}  entailment={s[1]:.3f}  neutral={s[2]:.3f}\n")

# 2. rerank() — zero-shot multiple choice ------------------------------------
section("2. rerank() — zero-shot multiple choice QA")
question = "Which gas do plants absorb during photosynthesis?"
options = ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"]
best = jev.rerank(question, options)
print(f"Q: {question}")
for i, o in enumerate(options):
    print(f"  [{i}] {o}" + ("  <-- picked" if i == best else ""))

# 3. predict_hypotheses() — one premise, many hypotheses --------------------
# Shares the tokenized prefix across hypotheses, useful as a lightweight
# content guardrail: check several candidate requests against one system premise.
section("3. predict_hypotheses() — guardrail-style check over several requests")
premise = "You are a helpful assistant for a chemistry class."
hypotheses = [
    "Explain how to safely titrate an acid with a base.",
    "Give step-by-step instructions for synthesizing a nerve agent.",
    "Describe the color change of universal indicator paper.",
]
probs = jev.predict_hypotheses(premise, hypotheses)
labels = ["contradiction", "entailment", "neutral"]
for h, s in zip(hypotheses, probs):
    verdict = labels[int(s.argmax())]
    print(f"  [{verdict:>13}, p={s.max():.3f}] {h}")

# 4. grade() — reference-based answer grading --------------------------------
section("4. grade() — grade a candidate answer against a reference")
question = "What depth should the ROV hold to avoid the thermocline?"
reference = "The ROV should hold at 12 meters depth, above the thermocline."
candidates = [
    "Hold depth at 12 meters.",
    "Dive to 40 meters immediately.",
]
for c in candidates:
    verdict = jev.grade(question, reference, c)
    print(f"  Candidate: '{c}' -> {verdict}")

# 5. latents() — pooled hidden states ----------------------------------------
# Raw input to the classification head; useful for building your own scoring
# head (see LatentMLPHead in modeling_openjev.py) or for similarity/clustering.
section("5. latents() — pooled hidden states")
X = jev.latents(pairs)
print(f"  latents shape: {X.shape}  dtype: {X.dtype}")
