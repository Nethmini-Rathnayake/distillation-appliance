# Appliance Distillation Prototype

A prototype of an on-premise "model factory" for air-gapped deployments at banks and
government agencies. The appliance serves a language model against customer-service
traffic, logs that traffic, distils a smaller specialist model from the logs, and proves
the specialist is faster, smaller, and no less accurate than what it replaced. Because
the box has a fixed memory bandwidth ceiling, shrinking the model is the only available
speed lever — distillation is a requirement, not an optimisation. The experiment runs
four arms over `banking77` (77 intents, ~13k real bank messages): a prompted teacher, a
prompted student, a student fine-tuned on gold labels, and the same student trained with
a KL term against the teacher's soft logits. Everything runs locally and offline after
the first model download.

## Status

Skeleton only. No model code yet — the modules under `factory/` are placeholders.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The first run downloads `roberta-large`, `distilbert-base-uncased`, and `banking77` into
`.cache/huggingface`. After that, set `runtime.offline: true` (the default) and the box
never reaches the network again.

## Running the arms

```bash
# A — teacher, few-shot prompted, no training
python -m factory.evaluate --config configs/arm_a.yaml

# B — student, same prompt, no training
python -m factory.evaluate --config configs/arm_b.yaml

# C — student fine-tuned on gold labels only
python -m factory.train    --config configs/arm_c.yaml

# D — same as C, plus KL against teacher soft logits
python -m factory.train    --config configs/arm_d.yaml
```

Arms A and B train nothing, so they go straight to evaluation. Arm D additionally needs
teacher soft logits over the train/val split — generate them once, before D:

```bash
python -m factory.teacher --config configs/arm_a.yaml --emit-logits
```

Each command writes `runs/<timestamp>_<arm>/` containing `config.yaml` (the fully
resolved config, not the arm file), `metrics.json`, and `predictions.parquet`.

**B vs D is the customer-demo number. C vs D is the research claim.**

## Configuration

All settings live in `configs/`. `base.yaml` holds everything shared; each arm file sets
`extends: base.yaml` and overrides only what makes that arm different. See
"Config inheritance" below.

## Reporting rules

- Every arm reports accuracy, macro F1, expected calibration error, p50 **and** p95
  latency, input token count, model size on disk, and peak memory.
- Latency is measured on the dev GPU and reported twice: `measured_*` and `adjusted_*`,
  where adjusted scales by `target_bandwidth / dev_bandwidth` (T4 → appliance = 0.85).
- The frozen eval set (`data/eval_frozen.parquet`) is carved before any training or
  teacher generation and is never touched again.
- If the teacher scores below `teacher.accuracy_floor`, the run stops and reports.
  Detecting a bad teacher cheaply is a product feature.

## Config inheritance

Each arm config declares `extends: base.yaml`. The loader reads the parent, then
recursively deep-merges the child over it: mappings merge key by key, and scalars and
lists are replaced wholesale. `extends` resolves relative to the config file's own
directory and is stripped from the merged result. `${a.b.c}` references are interpolated
against the merged config after the merge, so an arm can point at a shared path without
copying it.

The property that matters: an arm file contains only what makes that arm different.
`configs/arm_c.yaml` and `configs/arm_d.yaml` differ in the `loss` block and nothing
else, which is exactly the constraint the C vs D comparison rests on — `diff` on those
two files is the audit.
