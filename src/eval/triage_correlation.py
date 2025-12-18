"""Triage score correlation evaluation for clinical prioritization validation.

This module provides functions to evaluate the correlation and calibration of predicted
triage scores against actual clinical outcomes or ground truth severity scores.
"""

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_pearson_correlation(
    predicted_scores: NDArray[np.float64],
    actual_scores: NDArray[np.float64],
) -> tuple[float, float]:
    """Calculate Pearson correlation coefficient between predicted and actual scores.

    Args:
        predicted_scores: Array of predicted triage scores.
        actual_scores: Array of actual ground truth scores.

    Returns:
        Tuple of (correlation coefficient, p-value).
        Returns (nan, nan) if arrays are too short or have no variance.

    Raises:
        ValueError: If input shapes don't match.
    """
    if predicted_scores.shape != actual_scores.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted_scores.shape} vs actual {actual_scores.shape}")

    if len(predicted_scores) < 2:
        return (np.nan, np.nan)

    # Filter out non-finite values
    finite_mask = np.isfinite(predicted_scores) & np.isfinite(actual_scores)
    if np.sum(finite_mask) < 2:
        return (np.nan, np.nan)

    pred_finite = predicted_scores[finite_mask]
    actual_finite = actual_scores[finite_mask]

    # Check for zero variance
    if np.std(pred_finite) == 0 or np.std(actual_finite) == 0:
        return (np.nan, np.nan)

    # Compute correlation coefficient
    correlation = np.corrcoef(pred_finite, actual_finite)[0, 1]

    # Compute p-value using t-distribution
    n = len(pred_finite)
    t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))

    # Use simplified p-value approximation
    # For proper p-value, would use scipy.stats.t.sf(abs(t_stat), n-2) * 2
    # Here we use a simplified approach
    df = n - 2
    p_value = 2 * (1 - _t_cdf(abs(t_stat), df))

    return (float(correlation), float(p_value))


def compute_spearman_correlation(
    predicted_scores: NDArray[np.float64],
    actual_scores: NDArray[np.float64],
) -> tuple[float, float]:
    """Calculate Spearman rank correlation coefficient.

    Args:
        predicted_scores: Array of predicted triage scores.
        actual_scores: Array of actual ground truth scores.

    Returns:
        Tuple of (correlation coefficient, p-value).
        Returns (nan, nan) if arrays are too short.

    Raises:
        ValueError: If input shapes don't match.
    """
    if predicted_scores.shape != actual_scores.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted_scores.shape} vs actual {actual_scores.shape}")

    if len(predicted_scores) < 2:
        return (np.nan, np.nan)

    # Filter out non-finite values
    finite_mask = np.isfinite(predicted_scores) & np.isfinite(actual_scores)
    if np.sum(finite_mask) < 2:
        return (np.nan, np.nan)

    pred_finite = predicted_scores[finite_mask]
    actual_finite = actual_scores[finite_mask]

    # Convert to ranks
    pred_ranks = _rankdata(pred_finite)
    actual_ranks = _rankdata(actual_finite)

    # Use Pearson correlation on ranks
    return compute_pearson_correlation(pred_ranks, actual_ranks)


def compute_auc_roc(
    predicted_scores: NDArray[np.float64],
    binary_labels: NDArray[np.bool_],
) -> float:
    """Calculate Area Under the ROC Curve for binary classification.

    Args:
        predicted_scores: Array of predicted scores (higher = more positive).
        binary_labels: Array of binary ground truth labels (True = positive class).

    Returns:
        AUC-ROC value between 0 and 1. Returns nan if insufficient data.

    Raises:
        ValueError: If input shapes don't match or labels aren't binary.
    """
    if predicted_scores.shape != binary_labels.shape:
        raise ValueError(f"Shape mismatch: scores {predicted_scores.shape} vs labels {binary_labels.shape}")

    # Filter out non-finite scores
    finite_mask = np.isfinite(predicted_scores)
    pred_finite = predicted_scores[finite_mask]
    labels_finite = binary_labels[finite_mask]

    if len(pred_finite) < 2:
        return np.nan

    # Check if we have both classes
    n_positive = np.sum(labels_finite)
    n_negative = len(labels_finite) - n_positive

    if n_positive == 0 or n_negative == 0:
        return np.nan

    # Use Mann-Whitney U statistic to compute AUC
    # AUC = P(score_positive > score_negative)
    pos_scores = pred_finite[labels_finite]
    neg_scores = pred_finite[~labels_finite]

    # Count pairs where positive score > negative score
    # and handle ties (count as 0.5)
    u_stat = 0.0
    for pos_score in pos_scores:
        u_stat += np.sum(pos_score > neg_scores)
        u_stat += 0.5 * np.sum(pos_score == neg_scores)

    auc = u_stat / (n_positive * n_negative)
    return float(auc)


def compute_calibration_curve(
    predicted_probabilities: NDArray[np.float64],
    binary_labels: NDArray[np.bool_],
    n_bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Calculate calibration curve for probability predictions.

    Divides predictions into bins and computes the mean predicted probability
    and actual frequency of positive class in each bin.

    Args:
        predicted_probabilities: Array of predicted probabilities [0, 1].
        binary_labels: Array of binary ground truth labels (True = positive class).
        n_bins: Number of bins to divide predictions into.

    Returns:
        Tuple of (mean_predicted_prob, actual_frequency, bin_counts).
        - mean_predicted_prob: Mean predicted probability in each bin.
        - actual_frequency: Actual fraction of positives in each bin.
        - bin_counts: Number of samples in each bin.

    Raises:
        ValueError: If input shapes don't match or n_bins < 1.
    """
    if predicted_probabilities.shape != binary_labels.shape:
        raise ValueError(
            f"Shape mismatch: probabilities {predicted_probabilities.shape} vs labels {binary_labels.shape}"
        )

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    # Filter out non-finite probabilities
    finite_mask = np.isfinite(predicted_probabilities)
    probs_finite = predicted_probabilities[finite_mask]
    labels_finite = binary_labels[finite_mask]

    if len(probs_finite) == 0:
        return (np.array([]), np.array([]), np.array([]))

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs_finite, bin_edges[1:-1])

    mean_predicted = np.zeros(n_bins)
    actual_freq = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = bin_indices == i
        bin_counts[i] = np.sum(mask)

        if bin_counts[i] > 0:
            mean_predicted[i] = np.mean(probs_finite[mask])
            actual_freq[i] = np.mean(labels_finite[mask].astype(np.float64))
        else:
            mean_predicted[i] = np.nan
            actual_freq[i] = np.nan

    return (mean_predicted, actual_freq, bin_counts)


def calculate_expected_calibration_error(
    predicted_probabilities: NDArray[np.float64],
    binary_labels: NDArray[np.bool_],
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and actual frequencies,
    weighted by bin size.

    Args:
        predicted_probabilities: Array of predicted probabilities [0, 1].
        binary_labels: Array of binary ground truth labels (True = positive class).
        n_bins: Number of bins for calibration calculation.

    Returns:
        Expected Calibration Error value. Returns nan if no valid data.

    Raises:
        ValueError: If input shapes don't match or n_bins < 1.
    """
    mean_pred, actual_freq, bin_counts = compute_calibration_curve(predicted_probabilities, binary_labels, n_bins)

    if len(bin_counts) == 0 or np.sum(bin_counts) == 0:
        return np.nan

    # Compute weighted average of calibration errors
    ece = 0.0
    total_samples = np.sum(bin_counts)

    for i in range(len(bin_counts)):
        if bin_counts[i] > 0 and np.isfinite(mean_pred[i]) and np.isfinite(actual_freq[i]):
            weight = bin_counts[i] / total_samples
            ece += weight * abs(mean_pred[i] - actual_freq[i])

    return float(ece)


def _rankdata(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert data to ranks (average rank for ties).

    Args:
        data: Array of values to rank.

    Returns:
        Array of ranks (1-based, averaged for ties).
    """
    sorter = np.argsort(data)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(data))

    # Handle ties by using average rank
    sorted_data = data[sorter]
    obs = np.concatenate([[True], sorted_data[1:] != sorted_data[:-1]])
    dense = np.cumsum(obs)[inv]

    # Count number of elements with same rank
    count = np.concatenate([np.nonzero(obs)[0], [len(obs)]])
    ranks = 0.5 * (count[dense] + count[dense - 1] + 1)

    return np.asarray(ranks, dtype=np.float64)


def _t_cdf(t: float, df: int) -> float:
    """Approximate CDF of t-distribution (simplified version).

    This is a basic approximation. For production use, consider scipy.stats.t.cdf.

    Args:
        t: T-statistic value.
        df: Degrees of freedom.

    Returns:
        Approximate cumulative probability.
    """
    if df < 1:
        return np.nan

    # Use normal approximation for large df
    if df > 30:
        return _normal_cdf(t)

    # For small df, use a simple approximation
    # This is not highly accurate but avoids scipy dependency
    x = df / (df + t**2)
    # Incomplete beta function approximation (very simplified)
    # For better accuracy, use scipy.special.betainc(df/2, 0.5, x)
    prob = 1 - 0.5 * x ** (df / 2)

    if t < 0:
        prob = 1 - prob

    return float(np.clip(prob, 0, 1))


def _normal_cdf(x: float) -> float:
    """Approximate CDF of standard normal distribution.

    Uses error function approximation.

    Args:
        x: Value to evaluate CDF at.

    Returns:
        Approximate cumulative probability.
    """
    return float(0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi))))
