import torch
from modeling_openjev import OpenJevCrossEncoder

print(f"Loading OpenJEV onto {torch.cuda.get_device_name(0)}...")

# Load from local folder
jev = OpenJevCrossEncoder(".", subfolder="qwen3.5-4b-nli-v2")

# Test 1: Contradiction vs. Entailment
premise = "The vehicle is operating in shallow waters at 3 meters depth with rocky seabed."
hypotheses = [
    "The vehicle has sufficient clearance to descend another 10 meters.",  # Should contradict
    "The vehicle must maintain current depth to avoid grounding.",         # Should entail
]

scores = jev.predict([(premise, h) for h in hypotheses])
# Returns probabilities for: [contradiction, entailment, neutral]
for h, s in zip(hypotheses, scores):
    print(f"\nHypothesis: '{h}'")
    print(f"  Contradiction : {s[0]:.4f}")
    print(f"  Entailment    : {s[1]:.4f}")
    print(f"  Neutral       : {s[2]:.4f}")

# Test 2: Rerank Candidate Options
options = [
    "Descend 10 meters deeper.",
    "Maintain present depth.",
    "Full throttle forward into the seabed."
]
best_idx = jev.rerank(premise, options)
print(f"\nTop Recommended Action: {options[best_idx]}")