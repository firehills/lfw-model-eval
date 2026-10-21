import numpy as np
import pytest

from face_eval.metrics import score, verification_metrics


@pytest.mark.parametrize("metric,expected", [
    ("cosine", [1, 0]), ("dot", [4, 0]),
    ("euclidean", [0, np.sqrt(8)]), ("manhattan", [0, 4]),
])
def test_distance_definitions(metric, expected):
    np.testing.assert_allclose(score(np.array([2, 0]), np.array([[2, 0], [0, 2]]), metric), expected)


@pytest.mark.parametrize("metric,values,threshold", [
    ("cosine", [0.9, 0.2, 0.8, 0.1], 0.5),
    ("dot", [9, 2, 8, 1], 5),
    ("euclidean", [0.1, 0.8, 0.2, 0.9], 0.5),
    ("manhattan", [1, 8, 2, 9], 5),
])
def test_perfect_separation_and_threshold_direction(metric, values, threshold):
    result = verification_metrics(np.array([True, False, True, False]), np.array(values), threshold, metric)
    assert result["true_accepted"] == 2 and result["true_rejected"] == 2
    assert result["false_accepted"] == 0 and result["false_rejected"] == 0
    assert result["roc_auc"] == 1 and result["eer_interpolated"] == 0
    assert result["accuracy"] == result["f1"] == 1


def test_all_four_confusion_counts_and_rates():
    result = verification_metrics(
        np.array([1, 1, 0, 0]), np.array([0.9, 0.1, 0.8, 0.2]), 0.5, "cosine",
    )
    for name in ("true_accepted", "false_accepted", "false_rejected", "true_rejected"):
        assert result[name] == 1
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                 "false_accept_rate", "false_reject_rate"):
        assert result[name] == 0.5


@pytest.mark.parametrize("metric", ["cosine", "dot", "euclidean", "manhattan"])
def test_threshold_equality_is_accepted_and_tied_roc_is_unbiased(metric):
    result = verification_metrics(np.array([1, 0]), np.array([0.5, 0.5]), 0.5, metric)
    assert result["true_accepted"] == result["false_accepted"] == 1
    assert result["roc_auc"] == result["eer_interpolated"] == 0.5


def test_no_accepted_probes_and_invalid_scores():
    result = verification_metrics(np.array([1, 0]), np.array([0.9, 0.1]), 2, "cosine")
    assert result["precision"] == result["f1"] == 0
    with pytest.raises(ValueError, match="finite"):
        verification_metrics(np.array([1, 0]), np.array([np.nan, 0]), 0, "cosine")
    with pytest.raises(ValueError, match="Both genuine"):
        verification_metrics(np.array([1, 1]), np.array([1, 1]), 0, "cosine")
