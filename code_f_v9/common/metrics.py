from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_spearman(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 2 or np.nanstd(y) <= 1e-15 or np.nanstd(p) <= 1e-15:
        return np.nan
    return float(spearmanr(y, p, nan_policy="omit").statistic)


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {f"{prefix}n": 0, f"{prefix}rmse": np.nan, f"{prefix}mae": np.nan,
                f"{prefix}r2": np.nan, f"{prefix}spearman": np.nan, f"{prefix}mape": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p)) if len(y) >= 2 else np.nan
    sp = _safe_spearman(y, p)
    denom = np.where(np.abs(y) > 1e-12, np.abs(y), np.nan)
    mape = float(np.nanmean(np.abs(y-p)/denom)) if np.any(np.isfinite(denom)) else np.nan
    return {f"{prefix}n": int(len(y)), f"{prefix}rmse": rmse, f"{prefix}mae": mae,
            f"{prefix}r2": r2, f"{prefix}spearman": sp, f"{prefix}mape": mape}


def weighted_regression_metrics(y_true, y_pred, weight) -> dict:
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    w = np.asarray(weight, float)
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[mask], p[mask], w[mask]
    if len(y) == 0:
        return {"n": 0, "rmse_w": np.nan, "mae_w": np.nan, "spearman": np.nan}
    w = w / w.sum()
    rmse = float(np.sqrt(np.sum(w * (y-p)**2)))
    mae = float(np.sum(w * np.abs(y-p)))
    sp = _safe_spearman(y, p)
    return {"n": int(len(y)), "rmse_w": rmse, "mae_w": mae, "spearman": sp}


def pinball_loss(y_true, y_pred, tau: float = 0.95) -> float:
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return np.nan
    u = y[mask] - p[mask]
    return float(np.mean(np.maximum(tau*u, (tau-1)*u)))
