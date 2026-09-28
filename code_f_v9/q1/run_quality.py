from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common.io_utils import ensure_dir, read_csv, write_csv, write_json
from common.plotting import setup_style, savefig, auto_break_barplot
from common.random_utils import set_seed
from q1.quality_models import (
    QUALITY_FIELDS, load_directions, fit_quality_transform, transform_quality, critic_weights,
    adaptive_weights, pairwise_extreme_conflict, build_quality_scores, aggregate_domain_quality,
    bootstrap_domain_ci, expanded_vs_sample, perturbation_rank_stability,
)


def main():
    ap = argparse.ArgumentParser(description="Q1 quality evaluation: baselines + conflict-aware main model.")
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/q1_quality"))
    ap.add_argument("--directions", type=Path, default=Path(__file__).resolve().parents[1]/"config"/"quality_directions.json")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--critic-reference", choices=["all", "unique"], default="unique")
    ap.add_argument("--plot-sample", type=int, default=50000)
    ap.add_argument("--fast", action="store_true", help="Smoke-test settings: reduce bootstrap/sample size only; model definitions are unchanged.")
    args = ap.parse_args()
    if args.fast:
        args.bootstrap = min(args.bootstrap, 30)
        args.plot_sample = min(args.plot_sample, 5000)
    set_seed(args.seed); setup_style()
    out = ensure_dir(args.out); figs = ensure_dir(out/"figures")

    src = args.processed/"A"/"q1_quality.csv"
    df = read_csv(src, required=["record_id", "source_set", "quality_domain", *QUALITY_FIELDS])
    for c in QUALITY_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    directions = load_directions(args.directions)

    # Detect overlap. All rows are scored; by default duplicated record IDs count once when fitting global transforms/weights.
    overlap = (df.groupby("record_id")["source_set"].nunique().rename("n_source_sets").reset_index())
    overlap = overlap[overlap["n_source_sets"] > 1]
    write_csv(overlap, out/"q1_quality_record_overlap.csv")
    if args.critic_reference == "unique":
        priority = df["source_set"].map({"A2": 0, "A3": 0, "A1": 1}).fillna(2)
        tmp = df.assign(_priority=priority).sort_values(["record_id", "_priority"])
        ref_idx = tmp.drop_duplicates("record_id", keep="first").index
        fit_mask = df.index.isin(ref_idx)
    else:
        fit_mask = np.ones(len(df), dtype=bool)

    tr = fit_quality_transform(df, directions, fit_mask=fit_mask)
    z = transform_quality(df, tr)
    write_json({
        "directions": directions, "q01": tr.q01, "q99": tr.q99,
        "minimum": tr.minimum, "maximum": tr.maximum, "medians": tr.medians,
        "critic_reference": args.critic_reference,
    }, out/"q1_quality_transform.json")

    z_ref = z.loc[fit_mask]
    dom_ref = df.loc[fit_mask, "quality_domain"].fillna("unknown").astype(str)
    global_w, domain_w, weights_df, tau = adaptive_weights(z_ref, dom_ref)
    # Ensure every scored domain has a weight vector. For domains not present in ref (unlikely), use global.
    for d in df["quality_domain"].fillna("unknown").astype(str).unique():
        if str(d) not in domain_w:
            domain_w[str(d)] = global_w.copy()
    write_csv(weights_df, out/"q1_quality_weights.csv")
    write_json({"tau": tau, "global_weights": global_w.to_dict()}, out/"q1_quality_weight_meta.json")

    conflict_ref = 2.0*np.median(np.abs(z.loc[df["source_set"].eq("A1")].to_numpy(float) -
                                         np.median(z.loc[df["source_set"].eq("A1")].to_numpy(float), axis=1)[:,None]), axis=1)
    conflict_threshold = float(np.quantile(np.minimum(conflict_ref, 1.0), 0.95))
    scores = build_quality_scores(df, z, global_w, domain_w, conflict_threshold)
    write_csv(scores, out/"q1_sample_quality_scores.csv")

    corr = z_ref.corr(method="spearman")
    conflict_pair = pairwise_extreme_conflict(z_ref)
    write_csv(corr.reset_index().rename(columns={"index":"indicator"}), out/"q1_indicator_spearman.csv")
    write_csv(conflict_pair.reset_index().rename(columns={"index":"indicator"}), out/"q1_indicator_extreme_conflict.csv")

    methods = ["q_equal_mean", "q_global_critic_topsis", "q_adaptive_critic_topsis", "q_huber", "q_conflict_aware"]
    dom_tables = [aggregate_domain_quality(scores, m, weighted=(m=="q_conflict_aware")) for m in methods]
    domain_quality = pd.concat(dom_tables, ignore_index=True)
    boot_tables=[]
    exp_tables=[]
    for mi,m in enumerate(methods):
        bt=bootstrap_domain_ci(scores,m,args.bootstrap,args.seed+mi,weighted=(m=="q_conflict_aware")); boot_tables.append(bt)
        ex=expanded_vs_sample(scores,m); ex["method"]=m; exp_tables.append(ex)
    boot_all=pd.concat(boot_tables,ignore_index=True); domain_quality=domain_quality.merge(boot_all,on=["source_set","quality_domain","method","n"],how="left")
    write_csv(domain_quality, out/"q1_domain_quality_comparison.csv")
    expcmp=pd.concat(exp_tables,ignore_index=True) if exp_tables else pd.DataFrame(); write_csv(expcmp, out/"q1_expanded_vs_sample.csv")

    # Same perturbation applied to every method; weights stay fixed so this measures score robustness rather than re-estimation noise.
    perturb_rows=[]
    for trim in (0.01,0.05):
        lo=z.quantile(trim); hi=z.quantile(1-trim); zp=z.clip(lower=lo,upper=hi,axis=1)
        pert_scores=build_quality_scores(df,zp,global_w,domain_w,conflict_threshold)
        for m in methods:
            rho=float(pd.Series(scores[m]).corr(pd.Series(pert_scores[m]),method="spearman"))
            perturb_rows.append({"method":m,"perturbation":f"trim_{trim:.2f}","spearman_rank_stability":rho})
    perturb=pd.DataFrame(perturb_rows); write_csv(perturb, out/"q1_quality_perturbation_stability.csv")

    # E1/C0-C2 compact comparison table for the paper.
    score_rows=[]
    for m in methods:
        ex=expcmp[expcmp.method.eq(m)] if len(expcmp) else pd.DataFrame(); bt=boot_all[boot_all.method.eq(m)]; pt=perturb[perturb.method.eq(m)]
        score_rows.append({"method":m,
            "mean_abs_sample_expanded_delta":float(ex.delta.abs().mean()) if len(ex) else np.nan,
            "mean_bootstrap_cv":float(bt.q_boot_cv.mean()) if len(bt) else np.nan,
            "mean_perturbation_rank_spearman":float(pt.spearman_rank_stability.mean()) if len(pt) else np.nan,
            "explicit_conflict_handling":m in {"q_huber","q_conflict_aware"}})
    write_csv(pd.DataFrame(score_rows),out/"q1_quality_method_scorecard.csv")

    # Main-domain quality, one row per quality domain using expansion for arxiv/github and A1 for other domains.
    main_rows = []
    for dom, g in scores.groupby("quality_domain"):
        if dom == "arxiv" and (g["source_set"] == "A2").any():
            use = g[g["source_set"] == "A2"]
        elif dom == "github" and (g["source_set"] == "A3").any():
            use = g[g["source_set"] == "A3"]
        else:
            use = g[g["source_set"] == "A1"] if (g["source_set"] == "A1").any() else g
        q = float(np.sum(use["q_conflict_aware"]*use["reliability"])/(use["reliability"].sum()+1e-12))
        main_rows.append({"quality_domain": dom, "Q_domain": q, "n": len(use),
                          "mean_conflict": float(use["conflict_index"].mean()),
                          "high_conflict_rate": float(use["high_conflict"].mean()),
                          "source_used": ",".join(sorted(use["source_set"].astype(str).unique()))})
    main_domain = pd.DataFrame(main_rows).sort_values("Q_domain", ascending=False)
    write_csv(main_domain, out/"q1_domain_quality_main.csv")
    q0 = float(np.sum(scores["q_conflict_aware"]*scores["reliability"])/(scores["reliability"].sum()+1e-12))
    write_json({"Q0_global": q0, "conflict_threshold_A1_q95": conflict_threshold}, out/"q1_quality_summary.json")

    mapping_path = args.processed/"A"/"q1_domain_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        mixq = mapping.merge(main_domain[["quality_domain", "Q_domain"]], on="quality_domain", how="left")
        write_csv(mixq, out/"q1_mixture_domain_quality.csv")

    # Optional A18 statistics are not record-level joinable; use them as domain-level reasonableness diagnostics.
    text_path = args.processed/"A"/"q1_text_validation.csv"
    if text_path.exists():
        txt = pd.read_csv(text_path)
        if "domain" in txt.columns:
            stats = txt.groupby("domain", dropna=False).agg(
                n_text=("row_no", "count"), mean_chars=("text_char_count", "mean"),
                median_chars=("text_char_count", "median"), mean_tokens=("text_whitespace_token_count", "mean")
            ).reset_index()
            write_csv(stats, out/"q1_text_validation_domain_stats.csv")

    # Figures
    fig, ax = plt.subplots(figsize=(12, 10)); sns.heatmap(corr, cmap="vlag", center=0, ax=ax); ax.set_title("Spearman correlation among 22 quality indicators"); savefig(figs/"q1_01_indicator_spearman_heatmap.png", fig)
    topw = weights_df.groupby("indicator")["w_global"].first().sort_values()
    fig, axes, _ = auto_break_barplot(x=topw.values, y=topw.index, orient="h", color="#4C72B0", figsize=(8,7), title="Global CRITIC weights", xlabel="Weight"); savefig(figs/"q1_02_critic_weights.png", fig)
    fig, ax = plt.subplots(figsize=(12, 10)); sns.heatmap(conflict_pair, cmap="mako", ax=ax); ax.set_title("Extreme disagreement rate between indicator pairs"); savefig(figs/"q1_03_indicator_conflict_heatmap.png", fig)
    ps = scores.sample(min(args.plot_sample, len(scores)), random_state=args.seed)
    fig, ax = plt.subplots(figsize=(8, 6)); sns.scatterplot(data=ps, x="q_adaptive_critic_topsis", y="conflict_index", hue="quality_domain", s=18, alpha=.45, ax=ax); ax.axhline(conflict_threshold, ls="--", lw=1); ax.set_title("Quality consensus vs. conflict"); savefig(figs/"q1_04_quality_conflict_scatter.png", fig)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.violinplot(data=ps, x="quality_domain", y="q_conflict_aware", inner="quartile", cut=0, ax=ax); ax.tick_params(axis="x", rotation=35); ax.set_title("Conflict-aware quality distribution by domain"); savefig(figs/"q1_05_domain_quality_violin.png", fig)
    dq = main_domain.sort_values("Q_domain")
    fig, axes, _ = auto_break_barplot(data=dq, x="Q_domain", y="quality_domain", color="#4C72B0", figsize=(8,6), title="Main domain quality scores Q", xlim=(0,1)); savefig(figs/"q1_06_domain_quality_bar.png", fig)
    if len(expcmp):
        exmain=expcmp[expcmp.method.eq("q_conflict_aware")].copy()
        long = exmain.melt(id_vars="quality_domain", value_vars=["q_sample", "q_expanded"], var_name="dataset", value_name="Q")
        fig, axes, _ = auto_break_barplot(data=long, x="quality_domain", y="Q", hue="dataset", figsize=(7,5), title="A1 sample vs expanded-domain quality", ylim=(0,1)); savefig(figs/"q1_07_sample_vs_expanded.png", fig)
    method_dom = domain_quality.pivot_table(index="quality_domain", columns="method", values="domain_q", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6)); sns.heatmap(method_dom, annot=True, fmt=".3f", cmap="viridis", ax=ax); ax.set_title("Quality-method comparison by domain"); savefig(figs/"q1_08_quality_method_comparison.png", fig)

    print(f"Q1 quality complete: {out}")
    print(f"Q0={q0:.6f}; A1 conflict q95 threshold={conflict_threshold:.6f}; overlapping IDs={len(overlap)}")


if __name__ == "__main__":
    main()
