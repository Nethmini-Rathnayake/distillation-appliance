"""Synthetic fixtures.

Deliberately not banking77: these tests must run offline, in a second, and with
hand-checkable numbers. The real dataset is exercised by the arms, not here.
"""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset


def make_split(texts: list[str], labels: list[int]) -> Dataset:
    return Dataset.from_dict({"text": texts, "label": labels})


@pytest.fixture
def eval_split() -> Dataset:
    """A tiny frozen eval set: 4 examples over 2 labels."""
    return make_split(
        ["lost my card", "where is my refund", "card declined", "refund still pending"],
        [0, 1, 0, 1],
    )


@pytest.fixture
def train_split() -> Dataset:
    """A clean train pool sharing no text with `eval_split`.

    20 examples, 10 per label, so a 20% stratified carve is exactly 2 + 2.
    """
    texts = [f"train message {i}" for i in range(20)]
    labels = [i % 2 for i in range(20)]
    return make_split(texts, labels)


def probs_from_confidence(pairs: list[tuple[float, bool]]) -> tuple[np.ndarray, np.ndarray]:
    """Build 2-class probabilities with exact top-1 confidences.

    Each pair is (confidence, is_correct). The predicted class is always 0, so
    correctness is expressed through the gold label. Confidences must be >= 0.5,
    or class 0 would not be the argmax.
    """
    probs = np.array([[conf, 1.0 - conf] for conf, _ in pairs], dtype=np.float64)
    labels = np.array([0 if correct else 1 for _, correct in pairs], dtype=int)
    return probs, labels
