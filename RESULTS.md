# Results

> Fill this in after you actually run `train.py` and `evaluate.py`. Pull the
> numbers straight from `results/train_metrics.json` and
> `results/eval_results.json` — don't hand-type estimates. If you link a W&B
> report, put the link here too.

## Run configuration

- Model: `microsoft/Phi-3-mini-4k-instruct`
- Sample size: `<fill in>`
- Effective batch size: `<fill in>`
- Epochs: `<fill in>`
- Hardware: `<fill in, e.g. Kaggle T4 x1>`
- W&B report: `<link, if used>`

## Training

- Final training loss: `<from results/train_metrics.json>`
- Training time: `<from results/train_metrics.json>`
- Any incidents caught by `TrainingHealthCallback`? `<yes/no, details>`

## Safety evaluation

- Mean base toxicity: `<from results/eval_results.json>`
- Mean aligned toxicity: `<from results/eval_results.json>`

| Prompt (truncated) | Base toxicity | Aligned toxicity |
|---|---|---|
| ... | ... | ... |

## Reasoning sanity check

Paste 1-2 example completions from the aligned model here, so a reader can
judge qualitatively whether alignment training degraded coherence.

## Honest take

A couple of sentences on what these numbers do and don't demonstrate — e.g.
toxicity classifier limitations, small sample size, single epoch, whether
you'd trust this as a real safety signal or just a directional one.
