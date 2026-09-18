"""Evaluation against the frozen eval set.

    python -m factory.evaluate --config configs/arm_a.yaml

Also the entrypoint for the two untrained arms (A and B), which go straight from
a prompted forward pass to metrics.

Every arm reports the same metric block:
    accuracy, macro_f1, ece (expected calibration error),
    latency_p50_ms, latency_p95_ms, input_tokens,
    model_size_bytes, peak_memory_bytes

Rules:
    - Only the frozen test split is scored here. Never teacher-generated data.
    - Latency is reported as p50 and p95 separately; no single mean.
    - Calibration is a first-class metric, not a footnote: the escalation
      threshold in the product depends on it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

log = logging.getLogger(__name__)

DEFAULT_ECE_BINS = 10


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over `axis`."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Expected calibration error over `n_bins` equal-width confidence bins.

    Confidence is the top-1 probability. Each bin contributes
    |accuracy - mean confidence| weighted by its share of the examples; empty
    bins contribute nothing. Bins partition (0, 1] with the first bin closed at
    the bottom, so a confidence landing exactly on a boundary falls in the lower
    bin and every example is counted exactly once.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"probs has {probs.shape[0]} rows but labels has {labels.shape[0]}")
    if probs.shape[0] == 0:
        return 0.0

    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `digitize` with right=True puts x in bin i where edges[i-1] < x <= edges[i];
    # clipping folds confidence == 0.0 into the first bin rather than a bin of its own.
    bin_of = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    total = len(confidence)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_of == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = DEFAULT_ECE_BINS,
    label_names: Sequence[str] | None = None,
    num_labels: int | None = None,
) -> dict[str, Any]:
    """Accuracy, macro F1, per-class F1, and ECE from raw logits.

    Args:
        logits: (n_examples, n_classes) unnormalised scores.
        labels: (n_examples,) integer gold labels.
        n_bins: confidence bins for ECE.
        label_names: optional names, used to key `per_class_f1`. Defaults to the
            class index as a string.
        num_labels: class count to score over, so that a class absent from this
            eval slice still appears in `per_class_f1`. Defaults to the logit width.

    Never call this on teacher-generated data — the frozen eval set only.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (n_examples, n_classes), got shape {logits.shape}")
    labels = np.asarray(labels).astype(int)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"logits has {logits.shape[0]} rows but labels has {labels.shape[0]}")

    n_classes = num_labels if num_labels is not None else logits.shape[1]
    if label_names is not None and len(label_names) != n_classes:
        raise ValueError(f"got {len(label_names)} label names for {n_classes} classes")

    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    classes = list(range(n_classes))

    per_class = f1_score(labels, preds, labels=classes, average=None, zero_division=0)
    names = list(label_names) if label_names is not None else [str(c) for c in classes]

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=classes, average="macro", zero_division=0)),
        "per_class_f1": {name: float(score) for name, score in zip(names, per_class)},
        "ece": expected_calibration_error(probs, labels, n_bins=n_bins),
        "ece_bins": int(n_bins),
        "n_examples": int(len(labels)),
    }


def _cuda_sync() -> Callable[[], None] | None:
    """Return a CUDA synchronise callable, or None when CUDA is not in use.

    Without this, GPU work is timed to the point of dispatch rather than to
    completion and every latency number is fiction.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.synchronize


def measure_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    timed: int = 500,
    sync: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Run `warmup` untimed passes, then `timed` timed ones; report p50 and p95.

    Percentiles are reported separately and no mean is returned: on a voice or
    phone deployment it is the tail that breaks the product, and an average
    hides it.

    Args:
        fn: a zero-argument callable performing exactly one inference pass.
        warmup: untimed passes, to absorb allocation and kernel autotuning.
        timed: measured passes.
        sync: called after each pass before the clock is read. Defaults to
            `torch.cuda.synchronize` when CUDA is in use, else no-op.

    Returns:
        p50_ms, p95_ms, plus min/max and the pass counts for provenance.
    """
    if timed < 1:
        raise ValueError(f"timed must be >= 1, got {timed}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    if sync is None:
        sync = _cuda_sync()

    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    samples = np.empty(timed, dtype=np.float64)
    for i in range(timed):
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples[i] = (time.perf_counter() - start) * 1000.0

    return {
        "latency_p50_ms": float(np.percentile(samples, 50)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_min_ms": float(samples.min()),
        "latency_max_ms": float(samples.max()),
        "latency_warmup_passes": int(warmup),
        "latency_timed_passes": int(timed),
    }


class _NumpyEncoder(json.JSONEncoder):
    """Metrics arrive as numpy scalars often enough to be worth handling here."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_metrics(metrics: Mapping[str, Any], run_dir: str | Path) -> Path:
    """Write `metrics` to `<run_dir>/metrics.json`, creating the directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(dict(metrics), indent=2, sort_keys=True, cls=_NumpyEncoder) + "\n")
    log.info("wrote %s", path)
    return path
