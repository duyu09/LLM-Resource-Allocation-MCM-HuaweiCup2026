from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from common.compositional import ilr, aitchison_center, closure
from common.metrics import regression_metrics


def detect_columns(df: pd.DataFrame):
    p_cols = [c for c in df.columns if c.startswith("p_")]
    loss_cols = [c for c in df.columns if c.startswith("loss_") and c != "loss_nature"]
    if len(p_cols) != 17:
        raise ValueError(f"Expected 17 mixture columns, got {len(p_cols)}")
    if len(loss_cols) != 13:
        raise ValueError(f"Expected 13 loss columns, got {len(loss_cols)}")
    return p_cols, loss_cols


def first_order_features(P: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return np.asarray(P, float), [f"p{i}" for i in range(P.shape[1])]


def scheffe_features(P: np.ndarray, p_names: list[str] | None = None):
    P = np.asarray(P, float)
    n, d = P.shape
    names = p_names or [f"p{i}" for i in range(d)]
    cols = [P]
    fnames = list(names)
    pairs = []
    for i, j in itertools.combinations(range(d), 2):
        cols.append((P[:, i]*P[:, j])[:, None])
        fnames.append(f"{names[i]}*{names[j]}")
        pairs.append((i, j))
    return np.concatenate(cols, axis=1), fnames, pairs


@dataclass
class ScaledElasticNet:
    model: ElasticNetCV
    scaler: StandardScaler
    feature_names: list[str]

    def predict(self, X):
        return self.model.predict(self.scaler.transform(np.asarray(X, float)))

    @property
    def coef_(self):
        # Convert coefficients from the scaled feature space back to the original
        # Scheffe-feature scale. with_mean=False keeps the no-intercept mixture model.
        return self.model.coef_ / self.scaler.scale_


def fit_scaled_elastic_net(X, y, seed=42, cv=5):
    X = np.asarray(X, float); y = np.asarray(y, float)
    # Elastic Net is scale-sensitive. Do not center Scheffe mixture features because
    # the theoretical mixture model has no ordinary intercept; only rescale them.
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=np.logspace(-4, 0, 50),
        cv=cv, fit_intercept=False, max_iter=10000, tol=1e-6,
        random_state=seed, n_jobs=-1, selection="cyclic",
    ).fit(Xs, y)
    return ScaledElasticNet(model=model, scaler=scaler, feature_names=[])


def fit_models(train: pd.DataFrame, p_cols: list[str], loss_cols: list[str], seed=42):
    P = train[p_cols].to_numpy(float)
    X1, names1 = first_order_features(P)
    X2, names2, pairs = scheffe_features(P, p_cols)
    targets = loss_cols + ["loss_macro"]
    Y = train[loss_cols].copy()
    Y["loss_macro"] = Y.mean(axis=1)
    models = {"linear": {}, "scheffe": {}, "scheffe_en": {}}
    coef_rows = []
    for target in tqdm(targets, desc="Q1 mixture: fitting 14 loss targets", unit="target"):
        y = Y[target].to_numpy(float)
        m1 = LinearRegression(fit_intercept=False).fit(X1, y)
        m2 = LinearRegression(fit_intercept=False).fit(X2, y)
        m3 = fit_scaled_elastic_net(X2, y, seed=seed)
        m3.feature_names = names2
        models["linear"][target] = m1
        models["scheffe"][target] = m2
        models["scheffe_en"][target] = m3
        for method, coef, fn in [
            ("linear", m1.coef_, names1), ("scheffe", m2.coef_, names2), ("scheffe_en", m3.coef_, names2)
        ]:
            for name, value in zip(fn, coef):
                coef_rows.append({"method": method, "target": target, "feature": name, "coefficient": float(value)})
    return models, pd.DataFrame(coef_rows), names2, pairs


def predict_method(models, method: str, df: pd.DataFrame, p_cols, loss_cols):
    P = df[p_cols].to_numpy(float)
    X = P if method == "linear" else scheffe_features(P, p_cols)[0]
    out = pd.DataFrame(index=df.index)
    for target in loss_cols + ["loss_macro"]:
        out[target] = models[method][target].predict(X)
    return out


def evaluate_across_splits(panel: pd.DataFrame, models, p_cols, loss_cols):
    rows = []
    groups = list(panel.groupby(["pair_id", "split", "model_scale_label"], dropna=False))
    for (pair_id, split, scale), g in tqdm(groups, desc="Q1 mixture: evaluating splits", unit="split", leave=False):
        ymacro = g[loss_cols].mean(axis=1).to_numpy(float)
        for method in ["linear", "scheffe", "scheffe_en"]:
            pred = predict_method(models, method, g, p_cols, loss_cols)
            # Per-target + macro metrics. Absolute metrics are most meaningful at 1M; Spearman remains meaningful cross-scale.
            for target in loss_cols + ["loss_macro"]:
                y = ymacro if target == "loss_macro" else g[target].to_numpy(float)
                m = regression_metrics(y, pred[target].to_numpy(float))
                rows.append({"pair_id": pair_id, "split": split, "model_scale_label": scale,
                             "method": method, "target": target, **m})
    return pd.DataFrame(rows)


def fit_ilr_gp(train: pd.DataFrame, p_cols: list[str], seed=42):
    P = train[p_cols].to_numpy(float)
    X = ilr(P)
    y = train[[c for c in train.columns if c.startswith("loss_") and c != "loss_nature"]].mean(axis=1).to_numpy(float)
    kernel = ConstantKernel(1.0, (1e-3,1e3))*Matern(length_scale=np.ones(X.shape[1]), nu=1.5) + WhiteKernel(1e-4, (1e-8,1e0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=2)
    gp.fit(X, y)
    return gp


def evaluate_gp(panel: pd.DataFrame, gp, p_cols, loss_cols):
    rows = []
    groups = list(panel.groupby(["pair_id", "split", "model_scale_label"], dropna=False))
    for (pair_id, split, scale), g in tqdm(groups, desc="Q1 ILR-GP: evaluating splits", unit="split", leave=False):
        pred, std = gp.predict(ilr(g[p_cols].to_numpy(float)), return_std=True)
        y = g[loss_cols].mean(axis=1).to_numpy(float)
        m = regression_metrics(y, pred)
        rows.append({"pair_id": pair_id, "split": split, "model_scale_label": scale,
                     "method": "ilr_gp", "target": "loss_macro", **m,
                     "mean_pred_std": float(np.mean(std))})
    return pd.DataFrame(rows)


def interaction_matrix(coef_df: pd.DataFrame, p_cols: list[str], method="scheffe_en") -> pd.DataFrame:
    d = len(p_cols)
    mat = np.zeros((d,d), float)
    use = coef_df[(coef_df.method==method) & (coef_df.target!="loss_macro")]
    mean_coef = use.groupby("feature")["coefficient"].mean()
    for i,j in itertools.combinations(range(d),2):
        f = f"{p_cols[i]}*{p_cols[j]}"
        v = float(mean_coef.get(f, 0.0))
        mat[i,j]=mat[j,i]=v
    return pd.DataFrame(mat, index=p_cols, columns=p_cols)


def select_empirical_reference_mixture(train: pd.DataFrame, p_cols: list[str], loss_cols: list[str], model=None, method="scheffe_en"):
    """Select a fixed-p reference strictly from observed A4/A5 recipes.

    The previous implementation continuously optimized the fitted quadratic response
    surface over the entire simplex.  That permits the optimizer to move far outside
    the empirical support of the 512 RegMix recipes and can produce non-physical
    negative cross-entropy predictions.  For Q3's fixed-p *reference* strategy we do
    not need a global off-support optimum; we need a reproducible composition that is
    actually supported by Q1 observations.

    Therefore this function chooses the A4/A5 recipe with the smallest observed
    macro validation loss.  The selected p is an exact observed composition, so its
    support distance is zero and no extrapolative optimizer is involved.  The fitted
    main model is evaluated at that observed recipe only for a diagnostic prediction.
    """
    if train.empty:
        raise ValueError("Cannot select a reference mixture from an empty training set")
    P=train[p_cols].to_numpy(float)
    Y=train[loss_cols].to_numpy(float)
    macro=np.nanmean(Y,axis=1)
    valid=np.isfinite(macro) & np.all(np.isfinite(P),axis=1) & (P>=0).all(axis=1)
    if not np.any(valid):
        raise ValueError("No finite observed A4/A5 recipes are available for fixed-p selection")
    valid_idx=np.flatnonzero(valid)
    local=int(np.argmin(macro[valid]))
    pos=int(valid_idx[local])
    p=closure(P[pos],eps=0)
    observed=float(macro[pos])

    predicted=np.nan
    if model is not None:
        X=p.reshape(1,-1) if method=="linear" else scheffe_features(p.reshape(1,-1),p_cols)[0]
        predicted=float(model.predict(X)[0])

    source_index=train.iloc[pos]["index"] if "index" in train.columns else pos
    res=OptimizeResult(
        x=p.copy(),
        fun=predicted if np.isfinite(predicted) else observed,
        success=True,
        message="Selected best observed A4/A5 recipe; off-support continuous simplex optimization disabled.",
    )
    meta={
        "selection_method":"best_observed_A4_A5_macro_loss",
        "source_row_position":pos,
        "source_index":source_index.item() if hasattr(source_index,"item") else source_index,
        "observed_macro_loss_1M":observed,
        "predicted_macro_loss_1M":predicted,
        "candidate_count":int(np.sum(valid)),
        "support_distance":0.0,
    }
    return p, observed, predicted, res, meta


def optimize_mixture(*args, **kwargs):
    """Disabled legacy entry point.

    Unconstrained continuous optimization of the fitted Scheffe response over the
    whole simplex is deliberately not exposed because it can leave the observed
    recipe support and generate non-physical negative cross-entropy predictions.
    Use :func:`select_empirical_reference_mixture` for the Q3 fixed-p reference.
    """
    raise RuntimeError(
        "Legacy off-support simplex optimization is disabled. "
        "Use select_empirical_reference_mixture(train, p_cols, loss_cols, model)."
    )

