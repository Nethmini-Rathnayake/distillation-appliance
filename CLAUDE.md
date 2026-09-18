# Appliance Distillation Prototype

## What this project is

A prototype of an on-premise "model factory". It simulates an air-gapped appliance that:

1. Serves a language model to answer customer-service queries
2. Logs its own traffic
3. Distils a smaller specialist model from those logs
4. Proves the specialist is faster, smaller, and no less accurate

The end product this prototypes is a sealed box sold to banks and government agencies.

## Business context (read this before making design decisions)

- Customers are banks and government agencies on air-gapped networks.
- Their data never leaves their building. **The entire training loop must be able to
  run on the box.** Any design that requires phoning home is wrong.
- The box has a fixed memory bandwidth ceiling. Generation speed is roughly
  `memory_bandwidth / model_size_in_bytes`. You cannot buy more bandwidth.
- Therefore shrinking the model is the only available speed lever. Distillation is
  mandatory here, not an optimisation.

## Hardware we are simulating

| Profile        | Memory | Bandwidth  | Notes                      |
|----------------|--------|------------|----------------------------|
| `spark`        | 128 GB | 273 GB/s   | NVIDIA DGX Spark class     |
| `macmini`      | 64 GB  | 273 GB/s   | Mac mini M4 Pro class      |
| `macmini_small`| 16 GB  | 273 GB/s   | Entry appliance            |

Development runs on Colab or Kaggle T4 (16 GB, 320 GB/s). Measured tok/s on a T4
should be multiplied by `273/320 = 0.85` to estimate appliance performance. Always
report both the measured number and the adjusted estimate, clearly labelled.

## The core experiment

Four arms. Identical eval set, identical seed, identical base checkpoint for B/C/D.

| Arm | Name                | What it is                                              |
|-----|---------------------|---------------------------------------------------------|
| A   | `teacher_prompted`  | Large model, few-shot prompt, no training               |
| B   | `student_prompted`  | Small model, same prompt, no training                   |
| C   | `student_hard`      | Small model fine-tuned on gold labels only              |
| D   | `student_distilled` | Same as C, but loss adds KL against teacher soft logits |

**C vs D is the only pair that isolates distillation from fine-tuning.** Arms C and D
must differ in exactly one thing: the loss function. Same data, same epochs, same
learning rate, same seed. If you find yourself tuning them differently, stop.

B vs D is the number for the customer demo. C vs D is the number for the research claim.

## Distillation loss

```
loss = alpha * CrossEntropy(student_logits, hard_label)
     + (1 - alpha) * (T**2) * KLDiv(
           log_softmax(student_logits / T),
           softmax(teacher_logits / T)
       )
```

- `T` (temperature) flattens the teacher's distribution so the ranking of wrong answers
  becomes visible to the gradient. Start at T=3.
- The `T**2` factor rescales gradients so they stay comparable to the hard-label term.
- `alpha` blends the two. Start at 0.5.
- T and alpha are the two knobs. Sweep them; do not guess once and move on.

## Non-negotiable rules

- Carve out the eval set **before** any training or teacher generation. Never touch it again.
- Never evaluate on data produced by the teacher.
- Every arm reports: accuracy, macro F1, expected calibration error, p50 latency,
  p95 latency, input token count, model size on disk, peak memory.
- Calibration matters as much as accuracy. The product depends on the confidence score
  being trustworthy enough to trigger escalation. Measure it.
- If the teacher scores below the target accuracy bar, stop and report. Distilling from
  a bad teacher cannot work, and detecting that cheaply is a product feature.
- Everything must run offline after the first model download.

## Dataset

`banking77` from HuggingFace. 77 intents, ~13,000 real bank customer-service messages.
Chosen because it is literally the target customer's domain.

Split: use the provided train/test split. Carve 20% of train as validation. The provided
test set is the frozen eval set and is used only at the very end of each arm.

## Stack

Python 3.11, PyTorch, transformers, datasets, evaluate, peft, scikit-learn, pyyaml.
No cloud APIs. No web framework until the four arms produce numbers.

Teacher: an open model run locally (start with `roberta-large`). Do not use a hosted API
for the teacher — arms C and D need reproducible logits, and an API cannot give you those.

Student: `distilbert-base-uncased` to start.

## Conventions

- All configuration in YAML under `configs/`. Nothing hardcoded in scripts.
- Every run writes `runs/<timestamp>_<arm>/` containing `config.yaml`, `metrics.json`,
  and `predictions.parquet`.
- Scripts run standalone: `python -m factory.train --config configs/arm_d.yaml`
- Set every seed (python, numpy, torch, cuda) from the config. Reproducibility is the
  whole point of the experiment.

## What not to do

- Do not add a UI, API server, or dashboard until all four arms produce metrics.json.
- Do not use different hyperparameters for arms C and D.
- Do not silently fall back to CPU. If CUDA is unavailable, say so loudly in the log,
  because it invalidates every latency number.
- Do not report a single latency number. Report p50 and p95 separately; the tail is what
  breaks a voice or phone deployment.