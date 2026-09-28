from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import trange

QUALITY_FIELDS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "dsir_books", "dsir_wiki",
    "dsir_math", "qurater", "ad_en", "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_doc_unigram_entropy", "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction", "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]


@dataclass
class QualityTransform:
    q01: dict[str, float]
    q99: dict[str, float]
    minimum: dict[str, float]
    maximum: dict[str, float]
    directions: dict[str, int]
    medians: dict[str, float]


def load_directions(path: str | Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    missing = [c for c in QUALITY_FIELDS if c not in d]
    if missing:
        raise ValueError(f"Direction config missing fields: {missing}")
    out = {k: int(v) for k, v in d.items() if k in QUALITY_FIELDS}
    bad = [k for k, v in out.items() if v not in (-1, 1)]
    if bad:
        raise ValueError(f"Directions must be +/-1: {bad}")
    return out


def fit_quality_transform(df: pd.DataFrame, directions: dict[str, int], fit_mask=None) -> QualityTransform:
    ref = df.loc[fit_mask, QUALITY_FIELDS].copy() if fit_mask is not None else df[QUALITY_FIELDS].copy()
    ref = ref.apply(pd.to_numeric, errors="coerce")
    med = ref.median(numeric_only=True).to_dict()
    ref = ref.fillna(med)
    q01 = ref.quantile(0.01).to_dict()
    q99 = ref.quantile(0.99).to_dict()
    clipped = ref.clip(lower=pd.Series(q01), upper=pd.Series(q99), axis=1)
    minimum = clipped.min().to_dict()
    maximum = clipped.max().to_dict()
    return QualityTransform(q01, q99, minimum, maximum, directions, med)


def transform_quality(df: pd.DataFrame, tr: QualityTransform) -> pd.DataFrame:
    x = df[QUALITY_FIELDS].apply(pd.to_numeric, errors="coerce").copy()
    x = x.fillna(pd.Series(tr.medians))
    x = x.clip(lower=pd.Series(tr.q01), upper=pd.Series(tr.q99), axis=1)
    z = pd.DataFrame(index=df.index)
    eps = 1e-12
    for c in QUALITY_FIELDS:
        lo, hi = tr.minimum[c], tr.maximum[c]
        den = max(float(hi-lo), eps)
        if tr.directions[c] == 1:
            z[c] = (x[c]-lo)/den
        else:
            z[c] = (hi-x[c])/den
        z[c] = z[c].clip(0, 1)
    return z


def critic_weights(z: pd.DataFrame) -> pd.Series:
    arr = z[QUALITY_FIELDS].to_numpy(float)
    sigma = np.nanstd(arr, axis=0, ddof=1)
    corr = z[QUALITY_FIELDS].corr(method="spearman").to_numpy(float)
    corr = np.nan_to_num(corr, nan=0.0)
    info = sigma * np.sum(1.0 - np.abs(corr), axis=1)
    if not np.isfinite(info).all() or info.sum() <= 0:
        return pd.Series(np.ones(len(QUALITY_FIELDS))/len(QUALITY_FIELDS), index=QUALITY_FIELDS)
    return pd.Series(info/info.sum(), index=QUALITY_FIELDS)


def adaptive_weights(z: pd.DataFrame, domains: pd.Series, tau: float | None = None):
    global_w = critic_weights(z)
    counts = domains.value_counts(dropna=False)
    if tau is None:
        tau = float(counts.median()) if len(counts) else 1.0
    rows = []
    table = {}
    for d, idx in domains.groupby(domains).groups.items():
        zd = z.loc[idx, QUALITY_FIELDS]
        wd = critic_weights(zd) if len(zd) >= 10 else global_w.copy()
        n = len(zd)
        rho = n/(n+tau) if (n+tau)>0 else 0.0
        ws = rho*wd + (1-rho)*global_w
        ws = ws/ws.sum()
        table[str(d)] = ws
        for c in QUALITY_FIELDS:
            rows.append({"domain": d, "indicator": c, "n_domain": n, "rho_domain": rho,
                         "w_global": global_w[c], "w_domain": wd[c], "w_shrink": ws[c]})
    return global_w, table, pd.DataFrame(rows), tau


def topsis_scores(z: pd.DataFrame, weights: pd.Series | np.ndarray) -> np.ndarray:
    arr = z[QUALITY_FIELDS].to_numpy(float)
    w = np.asarray(weights, float).reshape(1, -1)
    dplus = np.sqrt(np.sum((w*(arr-1.0))**2, axis=1))
    dminus = np.sqrt(np.sum((w*arr)**2, axis=1))
    return dminus/(dplus+dminus+1e-12)


def adaptive_topsis_scores(z: pd.DataFrame, domains: pd.Series, weight_table: dict[str, pd.Series]) -> np.ndarray:
    out = np.empty(len(z), dtype=float)
    for d, idx in domains.groupby(domains).groups.items():
        loc = z.index.get_indexer(idx)
        w = weight_table.get(str(d))
        if w is None:
            w = pd.Series(np.ones(len(QUALITY_FIELDS))/len(QUALITY_FIELDS), index=QUALITY_FIELDS)
        out[loc] = topsis_scores(z.loc[idx], w)
    return out


def conflict_index(z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = z[QUALITY_FIELDS].to_numpy(float)
    med = np.median(arr, axis=1)
    mad = np.median(np.abs(arr-med[:, None]), axis=1)
    c = np.minimum(2.0*mad, 1.0)
    return c, med, mad


def huber_consensus(z: pd.DataFrame, domains: pd.Series, weight_table: dict[str, pd.Series],
                    c: float = 1.345, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    arr = z[QUALITY_FIELDS].to_numpy(float)
    W = np.zeros_like(arr)
    for d, idx in domains.groupby(domains).groups.items():
        loc = z.index.get_indexer(idx)
        w = weight_table[str(d)].reindex(QUALITY_FIELDS).to_numpy(float)
        W[loc, :] = w
    mu = np.sum(W*arr, axis=1)/(np.sum(W, axis=1)+1e-12)
    row_med = np.median(arr, axis=1)
    mad = np.median(np.abs(arr-row_med[:, None]), axis=1)
    scale = 1.4826*mad + 1e-8
    for _ in range(max_iter):
        r = (arr-mu[:, None])/scale[:, None]
        absr = np.abs(r)
        h = np.ones_like(r)
        mask = absr > c
        h[mask] = c/(absr[mask]+1e-12)
        eff = W*h
        new = np.sum(eff*arr, axis=1)/(np.sum(eff, axis=1)+1e-12)
        if np.max(np.abs(new-mu)) < tol:
            mu = new
            break
        mu = new
    return np.clip(mu, 0.0, 1.0)


def pairwise_extreme_conflict(z: pd.DataFrame, hi: float = 0.75, lo: float = 0.25) -> pd.DataFrame:
    a = z[QUALITY_FIELDS].to_numpy(float)
    p = len(QUALITY_FIELDS)
    out = np.zeros((p, p), float)
    high = a >= hi
    low = a <= lo
    for i in range(p):
        for j in range(i+1, p):
            v = np.mean((high[:, i] & low[:, j]) | (high[:, j] & low[:, i]))
            out[i, j] = out[j, i] = v
    return pd.DataFrame(out, index=QUALITY_FIELDS, columns=QUALITY_FIELDS)


def build_quality_scores(df: pd.DataFrame, z: pd.DataFrame, global_w: pd.Series,
                         domain_weights: dict[str, pd.Series], conflict_threshold: float):
    domains = df["quality_domain"].fillna("unknown").astype(str)
    equal = z[QUALITY_FIELDS].mean(axis=1).to_numpy(float)
    global_t = topsis_scores(z, global_w)
    adaptive_t = adaptive_topsis_scores(z, domains, domain_weights)
    c, _, mad = conflict_index(z)
    huber = huber_consensus(z, domains, domain_weights)
    final_q = np.clip((1-c)*adaptive_t + c*huber, 0.0, 1.0)
    out = df[[c for c in ["record_id", "source_set", "source_file", "quality_domain", "sub_path"] if c in df.columns]].copy()
    out["q_equal_mean"] = equal
    out["q_global_critic_topsis"] = global_t
    out["q_adaptive_critic_topsis"] = adaptive_t
    out["q_huber"] = huber
    out["conflict_mad"] = mad
    out["conflict_index"] = c
    out["high_conflict"] = c > conflict_threshold
    out["reliability"] = 1.0-c
    out["q_conflict_aware"] = final_q
    return out


def aggregate_domain_quality(scores: pd.DataFrame, q_col: str, weighted: bool = True) -> pd.DataFrame:
    rows = []
    for (src, dom), g in scores.groupby(["source_set", "quality_domain"], dropna=False):
        q = pd.to_numeric(g[q_col], errors="coerce").to_numpy(float)
        if weighted and "reliability" in g:
            w = pd.to_numeric(g["reliability"], errors="coerce").fillna(0).to_numpy(float)
            value = float(np.sum(w*q)/(np.sum(w)+1e-12))
        else:
            value = float(np.nanmean(q))
        rows.append({"source_set": src, "quality_domain": dom, "method": q_col,
                     "n": len(g), "domain_q": value,
                     "high_conflict_rate": float(g["high_conflict"].mean()) if "high_conflict" in g else np.nan,
                     "mean_conflict": float(g["conflict_index"].mean()) if "conflict_index" in g else np.nan})
    return pd.DataFrame(rows)


def bootstrap_domain_ci(scores: pd.DataFrame, q_col: str = "q_conflict_aware", n_boot: int = 1000,
                        seed: int = 42, weighted: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (src, dom), g in scores.groupby(["source_set", "quality_domain"], dropna=False):
        q = g[q_col].to_numpy(float)
        w = g["reliability"].to_numpy(float) if weighted else np.ones(len(g), dtype=float)
        n = len(g)
        vals = np.empty(n_boot, dtype=float)
        for b in trange(n_boot, desc=f"Q1 bootstrap {src}/{dom}", unit="rep", leave=False):
            idx = rng.integers(0, n, size=n)
            qb, wb = q[idx], w[idx]
            vals[b] = np.sum(qb*wb)/(np.sum(wb)+1e-12)
        rows.append({"source_set": src, "quality_domain": dom, "method": q_col, "n": n,
                     "q_boot_mean": float(np.mean(vals)), "q_boot_sd": float(np.std(vals, ddof=1)),
                     "q_ci_low": float(np.quantile(vals, 0.025)), "q_ci_high": float(np.quantile(vals, 0.975)),
                     "q_boot_cv": float(np.std(vals, ddof=1)/(abs(np.mean(vals))+1e-12))})
    return pd.DataFrame(rows)


def expanded_vs_sample(scores: pd.DataFrame, q_col: str = "q_conflict_aware") -> pd.DataFrame:
    rows = []
    for dom, ext_source in [("arxiv", "A2"), ("github", "A3")]:
        a = scores[(scores["source_set"] == "A1") & (scores["quality_domain"] == dom)]
        b = scores[(scores["source_set"] == ext_source) & (scores["quality_domain"] == dom)]
        if len(a) == 0 or len(b) == 0:
            continue
        def agg(g):
            return float(np.sum(g[q_col]*g["reliability"])/(g["reliability"].sum()+1e-12))
        qa, qb = agg(a), agg(b)
        rows.append({"quality_domain": dom, "sample_source": "A1", "expanded_source": ext_source,
                     "q_sample": qa, "q_expanded": qb, "delta": qb-qa,
                     "relative_delta": (qb-qa)/(abs(qa)+1e-12),
                     "sample_conflict_rate": float(a["high_conflict"].mean()),
                     "expanded_conflict_rate": float(b["high_conflict"].mean())})
    return pd.DataFrame(rows)


def perturbation_rank_stability(z: pd.DataFrame, base_score: np.ndarray, weights: pd.Series,
                                trims=(0.01, 0.05)) -> pd.DataFrame:
    rows = []
    base = np.asarray(base_score, float)
    for trim in trims:
        lo = z.quantile(trim)
        hi = z.quantile(1-trim)
        zp = z.clip(lower=lo, upper=hi, axis=1)
        q = topsis_scores(zp, weights)
        rho = spearmanr(base, q, nan_policy="omit").statistic
        rows.append({"perturbation": f"trim_{trim:.2f}", "spearman_rank_stability": float(rho)})
    return pd.DataFrame(rows)
