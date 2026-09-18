"""The leakage assertion: the guard on the one rule that cannot be walked back.

If a frozen-eval example reaches train or val, every number the experiment
produces is inflated and unrecoverable, so the failure has to be loud.
"""

from __future__ import annotations

import pytest

from factory.data import (
    EvalLeakageError,
    assert_no_leakage,
    build_splits,
    stratified_split,
)

from .conftest import make_split


def test_clean_splits_pass(train_split, eval_split):
    train, val = stratified_split(train_split, val_fraction=0.2, seed=1337)
    assert_no_leakage(train, val, eval_split)  # must not raise


def test_leak_into_train_is_detected(train_split, eval_split):
    leaked = make_split(
        list(train_split["text"]) + ["card declined"],
        list(train_split["label"]) + [0],
    )
    val = make_split(["clean val row"], [1])

    with pytest.raises(EvalLeakageError) as excinfo:
        assert_no_leakage(leaked, val, eval_split)

    message = str(excinfo.value)
    assert "train" in message
    assert "card declined" in message
    assert "1 example(s)" in message


def test_leak_into_val_is_detected(train_split, eval_split):
    val = make_split(["where is my refund"], [1])

    with pytest.raises(EvalLeakageError) as excinfo:
        assert_no_leakage(train_split, val, eval_split)

    assert "val" in str(excinfo.value)


def test_leak_survives_case_and_whitespace_differences(train_split, eval_split):
    """A reformatted duplicate is still a duplicate."""
    val = make_split(["  Card   Declined  "], [0])

    with pytest.raises(EvalLeakageError):
        assert_no_leakage(train_split, val, eval_split)


def test_all_leaked_examples_are_counted(train_split, eval_split):
    val = make_split(list(eval_split["text"]), list(eval_split["label"]))

    with pytest.raises(EvalLeakageError) as excinfo:
        assert_no_leakage(train_split, val, eval_split)

    assert "4 example(s)" in str(excinfo.value)


def test_build_splits_asserts_before_returning(eval_split):
    """The assertion is not opt-in: the normal split path runs it."""
    poisoned = make_split(
        [f"train message {i}" for i in range(10)] + ["lost my card", "card declined"],
        [i % 2 for i in range(10)] + [0, 0],
    )

    with pytest.raises(EvalLeakageError):
        build_splits(poisoned, eval_split, val_fraction=0.2, seed=1337)


def test_build_splits_shapes_and_labels(train_split, eval_split):
    splits = build_splits(train_split, eval_split, val_fraction=0.2, seed=1337)

    assert splits.sizes() == {"train": 16, "val": 4, "eval": 4}
    assert splits.labels == ["0", "1"]
    assert splits.eval[:] == eval_split[:]  # the frozen split is passed through untouched


def test_stratified_split_preserves_label_balance(train_split):
    train, val = stratified_split(train_split, val_fraction=0.2, seed=1337)

    assert sorted(val["label"]) == [0, 0, 1, 1]
    assert sorted(train["label"]) == [0] * 8 + [1] * 8


def test_split_is_deterministic_under_seed(train_split):
    first, _ = stratified_split(train_split, 0.2, seed=1337)
    same, _ = stratified_split(train_split, 0.2, seed=1337)
    other, _ = stratified_split(train_split, 0.2, seed=7)

    assert first["text"] == same["text"]
    assert first["text"] != other["text"]
