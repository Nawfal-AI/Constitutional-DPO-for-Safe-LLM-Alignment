# Engineering Notes

This file is for the things a README shouldn't hold: what actually happened
while building this, what tradeoffs I made and why, and what I'd do
differently with more time or compute. Fill in your own answers honestly —
specific, first-person detail here is worth far more than a table mapping
this repo to a job description.

## Design decisions

- **Why DPO instead of PPO/RLHF-with-reward-model?** `<your reasoning>`
- **Why 4-bit QLoRA instead of full fine-tuning?** `<your reasoning —
  presumably hardware constraints; be specific about what broke or wouldn't
  fit>`
- **Why this sample size / epoch count?** `<your reasoning>`

## What broke, and how I debugged it

`<This is the most valuable section for a reviewer with systems experience.
Concretely: did you hit an OOM? A NaN loss? A tokenizer mismatch between base
model and chat template? What did the traceback say, what did you try, what
actually fixed it?>`

## What I'd change with more compute/time

- Real multi-GPU training (`accelerate launch` + DeepSpeed/FSDP) instead of
  single-GPU gradient accumulation, and what I'd expect to change about the
  training dynamics.
- A larger, more adversarial safety eval set instead of 5 illustrative
  prompts — ideally with human review, not just an automated toxicity
  classifier.
- Held-out capability benchmarks (e.g. MMLU, TruthfulQA) to check whether
  safety training traded off general reasoning ability.
- A real reward-model-based RLHF comparison against DPO, to see whether the
  two methods actually converge to similar behavior on this dataset.

## Honest limitations of this project as a portfolio piece

`<Say plainly what this does and doesn't demonstrate. E.g.: "This shows I can
wire together a modern preference-tuning stack and reason about VRAM/hardware
constraints. It does not demonstrate distributed systems experience, since
everything here runs on one GPU.">`
