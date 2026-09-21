# OpenJEV - local setup & test scripts

This repo tracks **only the small stuff**: a README and two test scripts for running
[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) locally. It deliberately does
**not** contain the model weights, videos, or the upstream model code — those are large,
already hosted on Hugging Face, and gitignored here (see `.gitignore`) so they never get pushed.

`openjev` is Qwen3.5 turned into a cross-encoder that reads a premise and a hypothesis and
scores `[contradiction, entailment, neutral]`. That single primitive is enough to rerank
multiple-choice answers, grade a candidate answer against a reference, and act as a
lightweight content guardrail.

## What's in this repo

| File | Purpose |
|---|---|
| `test_jev.py` | Minimal smoke test — load the model, score two hypotheses, rerank 3 options. |
| `test_examples.py` | Broader usage tour — `predict`, `predict_hypotheses`, `rerank`, `grade`, `latents`. |
| `.gitignore` | Keeps the cloned model weights/code out of this repo. |

## Prerequisites

- Linux + NVIDIA GPU with a recent driver (CPU also works, just much slower).
- `git` and `git-lfs`.
- Python 3.10+ with `pip`.
- ~9 GB free disk for the recommended `qwen3.5-4b-nli-v2` checkpoint (the 35B checkpoint is
  much larger — only pull it if you actually need it).

## Setup on a new machine

**1. Clone the upstream model repo.** This gives you the weights, tokenizer, and the
`modeling_openjev.py` / `modeling_qwen35_moe_seqcls.py` code that the test scripts import.
A plain clone only pulls small LFS *pointer* files, not the multi-GB weights, so it's fast:

```bash
git clone https://huggingface.co/AlexWortega/openjev
cd openjev
```

**2. Install git-lfs and pull the checkpoint you need.** Fetching only `qwen3.5-4b-nli-v2/`
avoids downloading the other (much larger) checkpoints:

```bash
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install
git lfs pull --include="qwen3.5-4b-nli-v2/*"
```

**3. Drop this repo's files into that same `openjev/` folder** (clone this repo elsewhere and
copy, or `wget`/`scp` the three files over):

```bash
git clone <URL-of-this-repo> _tmp
cp _tmp/README.md _tmp/.gitignore _tmp/test_jev.py _tmp/test_examples.py .
rm -rf _tmp
```

**4. Install Python dependencies.** This system uses PEP 668 "externally managed" Python, so
either use `--break-system-packages` or a venv:

```bash
# Option A: user install (what was used here)
pip install --user --break-system-packages -U torch torchvision transformers accelerate

# Option B: isolated venv (cleaner, recommended for a fresh machine)
python3 -m venv .venv && source .venv/bin/activate
pip install -U torch torchvision transformers accelerate
```

**5. Verify the environment:**

```bash
python3 -c "
mods = ['torch', 'torchvision', 'transformers', 'accelerate']
for m in mods:
    try:
        mod = __import__(m)
        print(f'✓ {m:<14}: {getattr(mod, \"__version__\", \"installed\")}')
    except ImportError:
        print(f'✗ {m:<14}: NOT installed')

import torch
cuda_ok = torch.cuda.is_available()
print(f'\nCUDA Available : {cuda_ok}')
if cuda_ok:
    print(f'Device Name    : {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version   : {torch.version.cuda}')
"
```

**6. Run the tests:**

```bash
python3 test_jev.py
python3 test_examples.py
```

## Troubleshooting

- **`Exception: data did not match any variant of untagged enum ModelWrapper`** when loading
  `tokenizer.json` — your `tokenizers`/`transformers` install is too old for the checkpoint's
  tokenizer file. Fix:
  ```bash
  pip install --user --break-system-packages -U transformers tokenizers
  ```
  Verified working with `transformers 5.17.0` / `tokenizers 0.23.2`. Check versions with:
  ```bash
  python3 -c "import transformers, tokenizers; print(transformers.__version__, tokenizers.__version__)"
  ```
- **`error: externally-managed-environment`** from pip — add `--break-system-packages`, or use
  a venv (see step 4, option B).
- **Slow inference / kernel warnings about `causal_conv1d` or `flash-linear-attention`** — these
  are optional fused kernels; the model falls back to a plain PyTorch implementation
  automatically. Safe to ignore, or `pip install causal-conv1d flash-linear-attention` for speed.
- **Only need one checkpoint** — always use `git lfs pull --include="<subfolder>/*"` rather than
  a bare `git lfs pull`, which would download every checkpoint (tens of GB, including the 35B
  MoE model).

## API quick reference

```python
from modeling_openjev import OpenJevCrossEncoder
jev = OpenJevCrossEncoder(".", subfolder="qwen3.5-4b-nli-v2")

jev.predict([(premise, hypothesis), ...])               # -> [[p_contradiction, p_entailment, p_neutral], ...]
jev.predict_hypotheses(premise, [hyp1, hyp2, ...])       # one premise, many hypotheses, shared prefix
jev.rerank(question, [option1, option2, ...])            # -> index of best option (zero-shot MCQ)
jev.grade(question, reference_answer, candidate_answer)  # -> "contradiction" | "entailment" | "neutral"
jev.latents([(premise, hypothesis), ...])                # -> pooled hidden states, for custom heads
```

Full details: [AlexWortega/openjev model card](https://huggingface.co/AlexWortega/openjev).
