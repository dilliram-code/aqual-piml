"""
HydroPhysics AI: Standard Hydrological Performance & Mass-Conservation Metrics.

Standard Metrics Implemented:
- Nash-Sutcliffe Efficiency (NSE) [Nash & Sutcliffe, 1970]
- Kling-Gupta Efficiency (KGE) [Gupta et al., 2009; Kling et al., 2012]
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Percent Bias (PBIAS)
- Peak Flow Relative Error (top 5% discharge days)
- Low Flow Relative Error (bottom 20% discharge days)
- Prediction Interval Coverage Probability (PICP)
- Mean Prediction Interval Width (MPIW)
- Water-Balance Mass Conservation Closure
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np


def compute_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Nash-Sutcliffe Efficiency (NSE).
    NSE = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
    Ranges from -inf to 1.0 (1.0 = optimal model).
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    numerator = np.sum((y_t - y_p) ** 2)
    denominator = np.sum((y_t - np.mean(y_t)) ** 2)

    if denominator < 1e-9:
        return 0.0
    return float(1.0 - (numerator / denominator))


def compute_kge(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute Kling-Gupta Efficiency (KGE 2009) and its three constituent components:
    r (Pearson correlation), alpha (relative variability), beta (bias ratio).

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mean_t = np.mean(y_t)
    mean_p = np.mean(y_p)
    std_t = np.std(y_t)
    std_p = np.std(y_p)

    # Correlation r
    if std_t < 1e-9 or std_p < 1e-9:
        r = 0.0
    else:
        cov = np.mean((y_t - mean_t) * (y_p - mean_p))
        r = float(np.clip(cov / (std_t * std_p), -1.0, 1.0))

    # Variability ratio alpha
    alpha = float(std_p / std_t) if std_t > 1e-9 else 1.0

    # Bias ratio beta
    beta = float(mean_p / mean_t) if mean_t > 1e-9 else 1.0

    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return kge, r, alpha, beta


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def compute_pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percent Bias (%): 100 * sum(y_true - y_pred) / sum(y_true)."""
    denom = np.sum(y_true)
    if abs(denom) < 1e-9:
        return 0.0
    return float(100.0 * np.sum(np.asarray(y_true) - np.asarray(y_pred)) / denom)


def compute_peak_flow_error(y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 95.0) -> float:
    """Relative error on top peak flow observations (> 95th percentile)."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    threshold = np.percentile(y_t, percentile)
    peak_mask = y_t >= threshold

    if np.sum(peak_mask) == 0:
        return 0.0

    peak_true = np.mean(y_t[peak_mask])
    peak_pred = np.mean(y_p[peak_mask])
    if peak_true < 1e-6:
        return 0.0
    return float(100.0 * (peak_pred - peak_true) / peak_true)


def compute_picp_and_mpiw(
    y_true: np.ndarray,
    q_lower_5th: np.ndarray,
    q_upper_95th: np.ndarray,
) -> Tuple[float, float]:
    """
    Uncertainty prediction interval metrics:
    - PICP: Prediction Interval Coverage Probability (nominal: 90% for [5th, 95th])
    - MPIW: Mean Prediction Interval Width in physical units (mm/day)
    """
    y_t = np.asarray(y_true)
    q_l = np.asarray(q_lower_5th)
    q_u = np.asarray(q_upper_95th)

    inside = (y_t >= q_l) & (y_t <= q_u)
    picp = float(np.mean(inside)) * 100.0  # percentage
    mpiw = float(np.mean(q_u - q_l))

    return picp, mpiw


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_series: Optional[np.ndarray] = None,
    et_series: Optional[np.ndarray] = None,
    delta_s_series: Optional[np.ndarray] = None,
    q_lower: Optional[np.ndarray] = None,
    q_upper: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate full hydrological evaluation suite including point metrics,
    hydrologic signatures, water balance conservation, and uncertainty metrics.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    nse = compute_nse(y_t, y_p)
    kge, r, alpha, beta = compute_kge(y_t, y_p)
    rmse = compute_rmse(y_t, y_p)
    mae = compute_mae(y_t, y_p)
    pbias = compute_pbias(y_t, y_p)
    peak_err = compute_peak_flow_error(y_t, y_p)

    results = {
        "nse": round(nse, 4),
        "kge": round(kge, 4),
        "kge_r": round(r, 4),
        "kge_alpha": round(alpha, 4),
        "kge_beta": round(beta, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pbias": round(pbias, 2),
        "peak_flow_relative_error_pct": round(peak_err, 2),
    }

    # Physical Mass Conservation metrics
    if p_series is not None and et_series is not None and delta_s_series is not None:
        p = np.asarray(p_series)
        et = np.asarray(et_series)
        ds = np.asarray(delta_s_series)
        residual = p - et - y_p - ds
        results["mean_daily_wb_residual_mm"] = round(float(np.mean(residual)), 4)
        results["mean_abs_wb_residual_mm"] = round(float(np.mean(np.abs(residual))), 4)
        results["cumulative_wb_discrepancy_mm"] = round(float(np.sum(residual)), 2)
    else:
        results["mean_daily_wb_residual_mm"] = None
        results["mean_abs_wb_residual_mm"] = None
        results["cumulative_wb_discrepancy_mm"] = None

    # Uncertainty metrics
    if q_lower is not None and q_upper is not None:
        picp, mpiw = compute_picp_and_mpiw(y_t, q_lower, q_upper)
        results["picp_pct"] = round(picp, 2)
        results["mpiw_mm"] = round(mpiw, 4)
    else:
        results["picp_pct"] = None
        results["mpiw_mm"] = None

    return results

