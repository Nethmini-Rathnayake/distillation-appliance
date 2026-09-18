"""Dataset loading and splitting.

Responsibilities:
    - Load `banking77` from a local HF cache (offline after first download).
    - Freeze the provided test split as the eval set before anything else runs.
    - Carve 20% of train as validation, stratified by label, seeded from config.

Rule: the frozen eval set is written once and never regenerated. Nothing in this
module may return eval rows to a training or teacher-generation caller — that is
what `assert_no_leakage` exists to prove, and `load_splits` runs it on every call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

# Fallbacks used only when a key is absent from the config; configs/base.yaml is
# the real source of truth for all of them.
DEFAULTS: dict[str, Any] = {
    "name": "banking77",
    "train_split": "train",
    "test_split": "test",
    "val_fraction": 0.2,
    "text_field": "text",
    "label_field": "label",
    "seed": 1337,
}


class EvalLeakageError(AssertionError):
    """Raised when a frozen-eval example is also present in train or val.

    Subclasses AssertionError so a caller that catches broad assertion failures
    still trips, while `except EvalLeakageError` stays precise.
    """


@dataclass(frozen=True)
class Splits:
    """The three splits plus the label vocabulary.

    `eval` is the provided test split, frozen. It is scored once, at the end of
    an arm, and is never used for training, validation, or teacher generation.
    """

    train: Dataset
    val: Dataset
    eval: Dataset
    labels: list[str]

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "eval": len(self.eval)}


def _cfg(config: Mapping[str, Any] | None, key: str) -> Any:
    """Read `key` from the dataset block of a resolved config, else DEFAULTS.

    Accepts either a full config (`{"seed": ..., "dataset": {...}}`) or a bare
    dataset block, so callers are not forced to unpack before calling in.
    """
    if config is None:
        return DEFAULTS[key]
    if key == "seed":
        return config.get("seed", DEFAULTS["seed"])
    dataset_block = config.get("dataset", config)
    return dataset_block.get(key, DEFAULTS[key])


def _normalise(text: Any) -> str:
    """Identity key for an example: case- and whitespace-insensitive text.

    Leakage detection has to survive the cosmetic differences that creep in
    between splits, so compare on this rather than on the raw string.
    """
    return " ".join(str(text).split()).casefold()


def _keys(dataset: Dataset, text_field: str) -> list[str]:
    return [_normalise(text) for text in dataset[text_field]]


def assert_no_leakage(
    train: Dataset,
    val: Dataset,
    eval_set: Dataset,
    text_field: str = "text",
    *,
    max_report: int = 5,
) -> None:
    """Fail loudly if any frozen-eval example also appears in train or val.

    Raises:
        EvalLeakageError: naming the offending split, the overlap count, and up
            to `max_report` of the colliding texts.
    """
    eval_keys = set(_keys(eval_set, text_field))
    if not eval_keys:
        return

    for split_name, split in (("train", train), ("val", val)):
        overlap = eval_keys.intersection(_keys(split, text_field))
        if overlap:
            shown = sorted(overlap)[:max_report]
            more = len(overlap) - len(shown)
            suffix = f" (+{more} more)" if more > 0 else ""
            raise EvalLeakageError(
                f"Frozen eval set leaked into {split_name}: {len(overlap)} example(s) "
                f"appear in both. The eval set must be carved before any training or "
                f"teacher generation and never touched again. Offending texts: "
                f"{shown}{suffix}"
            )


def stratified_split(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
    label_field: str = "label",
) -> tuple[Dataset, Dataset]:
    """Split `dataset` into (train, val), stratified by `label_field`.

    Stratification is done over row indices with scikit-learn so it does not
    depend on the label column being a `ClassLabel`, and is fully determined by
    `seed`.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction!r}")

    labels = list(dataset[label_field])
    train_idx, val_idx = train_test_split(
        range(len(labels)),
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return dataset.select(sorted(train_idx)), dataset.select(sorted(val_idx))


def label_names(dataset: Dataset, label_field: str = "label") -> list[str]:
    """Label vocabulary, by name where the feature carries one, else by index."""
    feature = dataset.features.get(label_field)
    names = getattr(feature, "names", None)
    if names:
        return list(names)
    return [str(i) for i in sorted(set(dataset[label_field]))]


def freeze_eval_set(eval_set: Dataset, path: str | Path) -> Path:
    """Write the eval set to `path` once. An existing file is never rewritten.

    Returns the path either way, so a caller can read back exactly the rows a
    previous run scored.
    """
    path = Path(path)
    if path.exists():
        log.info("frozen eval set already present at %s; leaving it untouched", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    eval_set.to_parquet(str(path))
    log.info("froze %d eval examples to %s", len(eval_set), path)
    return path


def load_splits(
    config: Mapping[str, Any] | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> Splits:
    """Load banking77 and return train/val/eval splits plus the label list.

    The provided test split becomes the frozen eval set. 20% of train (or
    `dataset.val_fraction`) is carved off as validation, stratified by label and
    seeded from `seed`. Leakage is asserted before the splits are handed back.
    """
    name = _cfg(config, "name")
    text_field = _cfg(config, "text_field")
    label_field = _cfg(config, "label_field")
    seed = _cfg(config, "seed")

    raw = load_dataset(name, cache_dir=str(cache_dir) if cache_dir else None)
    return build_splits(
        raw[_cfg(config, "train_split")],
        raw[_cfg(config, "test_split")],
        val_fraction=_cfg(config, "val_fraction"),
        seed=seed,
        text_field=text_field,
        label_field=label_field,
    )


def build_splits(
    train_split: Dataset,
    test_split: Dataset,
    *,
    val_fraction: float = DEFAULTS["val_fraction"],
    seed: int = DEFAULTS["seed"],
    text_field: str = DEFAULTS["text_field"],
    label_field: str = DEFAULTS["label_field"],
) -> Splits:
    """Carve val out of `train_split` and freeze `test_split` as eval.

    Split-shaping logic lives here rather than in `load_splits` so it can be
    exercised without touching the network or the real dataset.
    """
    eval_set = test_split
    train, val = stratified_split(train_split, val_fraction, seed, label_field)

    assert_no_leakage(train, val, eval_set, text_field)

    splits = Splits(train=train, val=val, eval=eval_set, labels=label_names(eval_set, label_field))
    log.info("splits: %s over %d labels (seed=%d)", splits.sizes(), splits.num_labels, seed)
    return splits
