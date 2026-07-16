# Engineering Notes

## Design decisions

- **Why DPO instead of PPO/RLHF-with-reward-model?** DPO removes the need for a
  separate reward model and the sampling/rollout loop that PPO requires — it
  optimizes directly on preference pairs with a single model and a closed-form
  loss. On a single 16GB T4, PPO's requirement to hold a policy model, a value
  model, and a reward model in memory (or swap between them) simultaneously
  makes it impractical; DPO's simpler memory footprint made it the only
  realistic choice for this hardware.

- **Why 4-bit QLoRA instead of full fine-tuning?** Phi-3-mini is 3.8B
  parameters. Full fine-tuning in bf16 would need roughly 4x that in
  parameters + gradients + optimizer states (~60GB+ with Adam), far beyond a
  16GB T4. 4-bit quantization (bitsandbytes NF4) plus LoRA adapters (~9M
  trainable params) brought this down to something that actually fits, at the
  cost of a small quality gap versus full fine-tuning that I didn't attempt to
  measure here.

- **Why this sample size / epoch count?** 3,000 preference pairs / 1 epoch was
  sized around wall-clock time, not around what's ideal for alignment quality.
  A single T4 running DPO (which does forward/backward passes on chosen *and*
  rejected responses per step, roughly 3-4x the compute of ordinary SFT) takes
  about 5 hours for this sample size — already a lot for a portfolio run.
  Training longer or on more data would likely improve `rewards/accuracies`
  beyond the 0.579 this run achieved, at a proportional time cost.

## What broke, and how I debugged it

This project went through several real, unplanned failures — not scripted
demo problems — over its life on Kaggle/Colab:

1. **Notebook-kernel argparse crash.** Running the script's code inside a
   notebook cell (not as a standalone process) meant Colab/Kaggle's own
   injected `-f kernel.json` argument reached `argparse`, which errored on an
   argument it didn't define. Fixed by switching from `parse_args()` to
   `parse_known_args()`, and separately by moving to a `%%writefile` +
   `!python train.py` subprocess pattern, which avoids this whole class of
   problem by giving the script a clean `sys.argv`.

2. **Torch runtime corruption
   (`RuntimeError: THPDtypeType.tp_dict == nullptr`).** Kaggle/Colab preload a
   torch build matched to their GPU driver at kernel startup. Pinning a torch
   version in `requirements.txt` caused `pip install` to overwrite that
   preloaded build mid-session, corrupting the already-loaded C extension.
   No code-level fix resolves this once it happens — only a kernel/session
   restart does. Fixed by not pinning torch at all and instead relying on the
   platform's own working install.

3. **Dataset schema change (`KeyError: 'is_response_0_safer'`).** The
   PKU-SafeRLHF dataset on Hugging Face changed its column naming upstream —
   it now reports the safer response via an integer `safer_response_id`
   column rather than the old boolean column. This wasn't something I could
   have prevented; datasets maintained by third parties can change shape
   without notice. Fixed by checking for either column name and raising a
   clear error naming the available columns if neither is found, so a future
   schema change fails loudly instead of silently mis-labeling preferences.

4. **`trust_remote_code` / rope_scaling incompatibility.** Loading Phi-3 with
   `trust_remote_code=True` pulls Microsoft's own cached modeling code, which
   was written against an older `rope_scaling` config schema (key `"type"`).
   Current `transformers` normalizes configs to a newer schema (key
   `"rope_type"`) before handing them to model code, so the cached remote code
   threw `KeyError: 'type'`. The same remote code also didn't support `sdpa`
   attention, which is the only backend that runs on a T4 (flash-attention-2
   needs Ampere+). Fixed by dropping `trust_remote_code` entirely — Phi-3 has
   had native support in `transformers` since 4.41, and the native
   implementation stays in sync with the library's own config parsing.

5. **`trl` API change (`DPOConfig` no longer accepts `max_prompt_length`).**
   An upstream version bump in `trl` dropped this parameter in favor of a
   single `max_length`. Fixed by removing the deprecated argument.

6. **Self-shadowing import (`AttributeError: module 'evaluate' has no
   attribute 'load'`).** My evaluation script was named `evaluate.py` — the
   same name as the Hugging Face `evaluate` package it needed to import.
   Because the running script's own directory is first on `sys.path`, `import
   evaluate` resolved to the script itself rather than the installed package,
   regardless of where in the file the import statement lived (a local import
   inside a function doesn't avoid this — the collision is based on the
   filename, not the import's location). Fixed by renaming the file to
   `eval_harness.py`.

Each of these (except #6, which was mine) was an upstream library or dataset
change, not a mistake in the original approach — which is itself a realistic
picture of what "debugging complex training issues" looks like in practice:
most of the time isn't wrong code, it's a moving dependency surface.

## What I'd change with more compute/time

- Real multi-GPU training (`accelerate launch` + DeepSpeed/FSDP) instead of
  single-GPU gradient accumulation — I'd expect this to mainly change
  wall-clock time rather than final model quality at this sample size, but
  it would let me scale to a larger sample without a multi-hour single-GPU
  run.
- A larger, more adversarial safety eval set. The 5-prompt eval here used
  fairly direct requests that Phi-3-mini-4k-**instruct** already handled well
  before any DPO training (both base and aligned toxicity scores were close
  to the classifier's floor) — a jailbreak-style prompt set would be a much
  more informative test of whether DPO training closes a real gap.
- Held-out capability benchmarks (MMLU, TruthfulQA) to check whether safety
  training traded off general reasoning ability, rather than relying on two
  spot-checked reasoning prompts.
- A real reward-model-based RLHF comparison against DPO on the same dataset,
  to see whether the two methods converge to similar behavior here.

## Honest limitations of this project as a portfolio piece

This demonstrates that I can wire together a modern preference-tuning stack
(QLoRA + DPO via TRL), reason about hardware constraints on a memory-limited
GPU, and debug real breakages across a fast-moving dependency stack (dataset
schema, `transformers`, `trl`). It does not demonstrate distributed systems
experience — everything here runs on one GPU — and the safety improvement
shown is real but modest, on an already-safety-tuned base model, using a
small and not particularly adversarial eval set.
