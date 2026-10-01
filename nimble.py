import requests

URL = "http://localhost:11434/v1/systemone"
MODEL = "nimble"


def ask(state, questions):
    r = requests.post(
        URL,
        json={"model": MODEL, "state": state, "questions": questions},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["answers"]


def show_choice(title, ans):
    print(f"\n=== {title} ===")
    print(f"Best hypothesis: {ans['choice']}  (confidence {ans['confidence']:.2f})")
    for k, p in sorted(ans["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {k:<14} {p:.3f}  {'#' * int(p * 30)}")


def show_bool(title, ans, threshold=0.5):
    p = ans["noul"]
    print(f"\n=== {title} ===")
    print(f"P(true) = {p:.3f}  ->  {'TRUE' if p >= threshold else 'FALSE'}")


# ---------- Multi-hypothesis (probability distribution) ----------

# 1) Which fault explains the symptoms?
a = ask(
    "Motor draws high current, vehicle drifts to the left, and there is a "
    "grinding noise from the left thruster.",
    {"fault": {
        "type": "choice",
        "instructions": "What is the most likely cause?",
        "criteria": {
            "mechanical_damage": "Physical damage or obstruction in a thruster",
            "sensor_failure": "A sensor is giving wrong readings",
            "software_bug": "Control software error",
            "battery_low": "Power supply problem",
        }}},
)
show_choice("Fault diagnosis", a["fault"])

# 2) Sentiment as a 3-way hypothesis
a = ask(
    "The simulation ran fine, but the results were not as good as we hoped.",
    {"sentiment": {
        "type": "choice",
        "instructions": "What is the overall sentiment?",
        "criteria": {"positive": None, "neutral": None, "negative": None},
    }},
)
show_choice("Sentiment", a["sentiment"])

# 3) Several questions about the same text in one call
a = ask(
    "Enemy armor column spotted moving north at dawn, fuel status unknown.",
    {
        "domain": {
            "type": "choice",
            "instructions": "Which category does this report belong to?",
            "criteria": {"intelligence": None, "logistics": None, "medical": None},
        },
        "priority": {
            "type": "choice",
            "instructions": "How should this be prioritized?",
            "criteria": {"low": None, "medium": None, "high": None},
        },
    },
)
show_choice("Report category", a["domain"])
show_choice("Report priority", a["priority"])

# ---------- Binary true/false ----------

# 4) Single boolean with a contrastive pair (one fact flips the answer)
policy = "Only Mira may authorize refunds for account 42. "
for signer in ("Mira", "Noah"):
    a = ask(
        policy + f"The refund for account 42 was signed by {signer}.",
        {"authorized": {"type": "noul",
                        "instructions": "Is the refund properly authorized?"}},
    )
    show_bool(f"Refund signed by {signer}", a["authorized"])

# 5) Boolean fact checks against a passage, with custom true/false descriptions
passage = ("The AUV completed 14 of 15 planned survey lines. "
           "Line 9 was aborted because of strong currents.")
a = ask(
    passage,
    {
        "all_done": {
            "type": "noul",
            "instructions": "Were all planned survey lines completed?",
            "criteria": {"true": "Every planned line was finished",
                         "false": "At least one line was not finished"},
        },
        "weather_issue": {
            "type": "noul",
            "instructions": "Did environmental conditions affect the mission?",
        },
    },
)
show_bool("All lines completed?", a["all_done"])
show_bool("Environmental impact?", a["weather_issue"])