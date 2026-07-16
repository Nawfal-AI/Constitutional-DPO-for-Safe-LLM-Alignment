# DPO Safety Alignment: Preference-Based Fine-Tuning on a Single GPU

![Kaggle](https://img.shields.io/badge/Kaggle-T4_GPU-blue)

A small end-to-end pipeline for aligning an open-weight LLM (Phi-3-mini) toward
safer responses using **Direct Preference Optimization (DPO)** on the
[PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)
preference dataset, built to run within a single 16GB T4 GPU (e.g. free-tier
Kaggle).

## What this is (and isn't)

This project trains a model on human-labeled *safe vs. unsafe* response pairs
using DPO. It is **not** an implementation of Constitutional AI — Constitutional
AI specifically refers to a method where a model critiques and revises its own
outputs against a written constitution, and those self-generated
critique/revision pairs (rather than only human labels) are used to train a
preference or reward model. This project is more limited in scope: standard
DPO on an existing human-preference dataset. I'm calling it what it is rather
than what it isn't.

## Repository structure

```
├── train.py             # DPO training: 4-bit QLoRA + health-monitoring callback + data validation
├── eval_harness.py       # Compares base vs. aligned model on toxicity + a reasoning sanity check
│                          # (named eval_harness.py, not evaluate.py, deliberately — a file named
│                          #  evaluate.py shadows the pip `evaluate` package on import and crashes)
├── requirements.txt      # Dependencies
└── results/              # Populated after you run train.py / eval_harness.py (see below)
```

## What the code actually does

| Component | What it does | What it doesn't do |
|---|---|---|
| `TrainingHealthCallback` | Watches loss each logging step; on NaN/Inf/spike, saves an emergency checkpoint and halts the run | It's a single-process callback, not a distributed monitoring/alerting system |
| `DataValidator` | Drops empty, too-short, or duplicate (chosen==rejected) preference pairs before training | It's a length/dedup filter, not a learned quality model |
| 4-bit QLoRA | Fits a 3.8B-parameter model's fine-tuning into 16GB VRAM | Full fine-tuning of larger models still needs more/bigger GPUs |
| `gradient_accumulation_steps` | Increases the *effective batch size* on a single GPU by accumulating gradients before stepping the optimizer | This is **not** multi-GPU or multi-node training — real distributed training would need something like `accelerate launch` across multiple devices with DeepSpeed/FSDP, which this repo doesn't implement |
| Attention implementation | Auto-detects GPU compute capability and uses `sdpa` on T4 (`flash_attention_2` requires Ampere+ and will not run on a T4) | — |

## How to run

### Kaggle (recommended free option)
1. New Kaggle Notebook, accelerator: **GPU T4 x2**, internet on.
2. Upload `train.py`, `eval_harness.py`, `requirements.txt`.
3. `pip install -r requirements.txt`
4. `python train.py --sample_size 3000` (add `--wandb_key YOUR_KEY` for live logging).
   On a single T4 this takes **several hours** (~5h for 3,000 samples/1 epoch in
   practice — DPO runs 3-4x the forward/backward compute of ordinary fine-tuning
   per step). Use `--sample_size 1000 --disable_gradient_checkpointing` for a
   much faster (~1-1.5h) iteration run.
5. `python eval_harness.py`

### Any single-GPU machine
Same steps locally, given a CUDA GPU with ≥16GB VRAM and the deps installed.

### Arguments (`train.py`)
| Argument | Default | Description |
|---|---|---|
| `--model_name` | `microsoft/Phi-3-mini-4k-instruct` | Base model |
| `--sample_size` | `5000` | Number of preference pairs to train on |
| `--effective_batch_size` | `16` | Target effective batch size (via grad accumulation) |
| `--per_device_train_batch_size` | `2` | Micro-batch size per step |
| `--wandb_key` | `None` | Enables W&B logging if set |

## Results

Results are written to `results/train_metrics.json` and
`results/eval_results.json` after each script runs — real numbers from an
actual run, not hardcoded here. See [`RESULTS.md`](RESULTS.md) for the most
recent run I did, including caveats on what the toxicity metric does and
doesn't show.

## Known limitations

- **Toxicity scoring is a proxy, not a safety guarantee.** The classifier used
  in `eval_harness.py` is a general-purpose toxicity model, not a substitute for
  human review or structured adversarial red-teaming. 5 illustrative prompts
  is not a benchmark.
- **Single-GPU only.** No multi-node/distributed training is implemented here.
- **1 epoch, small sample.** This is sized to run in ~25–30 minutes on a free
  Kaggle T4, not to produce a maximally-aligned model.

See [`NOTES.md`](NOTES.md) for a more detailed engineering writeup — what I'd
change with more compute/time, and what I learned running this.
