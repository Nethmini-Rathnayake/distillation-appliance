"""Expected calibration error, checked against hand-computed values.

Calibration is what the escalation threshold rests on, so ECE is verified
arithmetically rather than by regression against whatever the code first
happened to print.
"""

from __future__ import annotations

import numpy as np
import pytest

from factory.evaluate import DEFAULT_ECE_BINS, compute_metrics, expected_calibration_error

from .conftest import probs_from_confidence


def test_default_is_ten_bins():
    assert DEFAULT_ECE_BINS == 10


def test_perfect_calibration_is_zero():
    """Confidence 0.9 on 10 predictions, 9 of them correct: |0.9 - 0.9| = 0."""
    probs, labels = probs_from_confidence([(0.9, True)] * 9 + [(0.9, False)])

    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_total_overconfidence_equals_the_error_rate():
    """Confidence 1.0 everywhere, half of them wrong: ECE = |0.5 - 1.0| = 0.5."""
    probs, labels = probs_from_confidence([(1.0, True)] * 5 + [(1.0, False)] * 5)

    assert expected_calibration_error(probs, labels) == pytest.approx(0.5)


def test_underconfidence_counts_the_same_as_overconfidence():
    """ECE is a gap, not a signed error: 10 correct at confidence 0.6 is 0.4 off."""
    probs, labels = probs_from_confidence([(0.6, True)] * 10)

    assert expected_calibration_error(probs, labels) == pytest.approx(0.4)


def test_two_populated_bins_are_weighted_by_share():
    """4 @ conf 0.95, half right; 6 @ conf 0.65, all right.

    bin (0.9, 0.95]  -> (4/10) * |0.50 - 0.95| = 0.18
    bin (0.6, 0.7]   -> (6/10) * |1.00 - 0.65| = 0.21
    total                                        0.39
    """
    probs, labels = probs_from_confidence(
        [(0.95, True), (0.95, True), (0.95, False), (0.95, False)] + [(0.65, True)] * 6
    )

    assert expected_calibration_error(probs, labels) == pytest.approx(0.39)


def test_empty_bins_contribute_nothing():
    """Two examples in one bin score the same as a run with 8 bins left empty."""
    probs, labels = probs_from_confidence([(0.85, True), (0.85, False)])

    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.35)


def test_one_bin_collapses_to_the_global_gap():
    probs, labels = probs_from_confidence([(0.9, True), (0.7, False)])
    # mean confidence 0.8, accuracy 0.5
    assert expected_calibration_error(probs, labels, n_bins=1) == pytest.approx(0.3)


def test_bin_count_changes_the_estimate():
    """Splitting a mixed bin exposes error that a coarse binning averages away."""
    probs, labels = probs_from_confidence([(0.55, True)] * 10 + [(0.95, False)] * 10)

    coarse = expected_calibration_error(probs, labels, n_bins=1)  # |0.5 - 0.75|
    fine = expected_calibration_error(probs, labels, n_bins=10)  # 0.5*0.45 + 0.5*0.95

    assert coarse == pytest.approx(0.25)
    assert fine == pytest.approx(0.70)
    assert fine > coarse


def test_confidence_on_a_bin_edge_lands_in_the_lower_bin():
    """0.7 belongs to (0.6, 0.7], not (0.7, 0.8] — and is counted exactly once."""
    probs, labels = probs_from_confidence([(0.7, True), (0.7, False)])
    # Either way the value is |0.5 - 0.7|; what matters is that both examples
    # share a bin, which a double-counted or dropped example would break.
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.2)


def test_low_confidence_multiclass():
    """Uniform 3-way probabilities: confidence 1/3, accuracy 1/3, ECE 0."""
    probs = np.full((6, 3), 1.0 / 3.0)
    labels = np.array([0, 0, 1, 1, 2, 2])

    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_empty_input_is_zero():
    assert expected_calibration_error(np.zeros((0, 3)), np.zeros(0, dtype=int)) == 0.0


def test_rejects_mismatched_lengths():
    probs, labels = probs_from_confidence([(0.9, True)] * 3)

    with pytest.raises(ValueError, match="rows"):
        expected_calibration_error(probs, labels[:2])


def test_rejects_non_positive_bin_count():
    probs, labels = probs_from_confidence([(0.9, True)])

    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(probs, labels, n_bins=0)


def test_compute_metrics_reports_ece_over_ten_bins():
    """The metric block carries the same ECE the standalone function computes."""
    # log-probs recover the exact confidences through softmax.
    probs, labels = probs_from_confidence([(1.0 - 1e-12, True)] * 5 + [(1.0 - 1e-12, False)] * 5)
    logits = np.log(probs)

    metrics = compute_metrics(logits, labels)

    assert metrics["ece"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["ece_bins"] == 10
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["n_examples"] == 10


def test_compute_metrics_scores_every_class_including_absent_ones():
    logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 0.0, 0.0]])
    labels = np.array([0, 1, 0])

    metrics = compute_metrics(logits, labels, num_labels=3, label_names=["a", "b", "c"])

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["per_class_f1"] == {"a": 1.0, "b": 1.0, "c": 0.0}
    # class "c" never appears, so it drags macro F1 down — which is the point of
    # reporting macro over 77 intents rather than micro.
    assert metrics["macro_f1"] == pytest.approx(2.0 / 3.0)
