from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common.io_utils import ensure_dir, read_csv, write_csv, dump_joblib, write_json
from common.plotting import setup_style, savefig, auto_break_barplot
from common.random_utils import set_seed
from q1.mixture_models import detect_columns, fit_models, evaluate_across_splits, fit_ilr_gp, evaluate_gp, interaction_matrix, select_empirical_reference_mixture, predict_method


def main():
    ap = argparse.ArgumentParser(description="Q1 mixture modeling: Linear / Scheffe / ElasticNet / ILR-GP.")
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/q1_mixture"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-gp", action="store_true")
    ap.add_argument("--fast", action="store_true", help="Smoke-test mode. Keeps all deterministic mixture baselines; skips the cubic-cost GP fit.")
    args = ap.parse_args()
    if args.fast:
        args.skip_gp = True
    set_seed(args.seed); setup_style()
    out=ensure_dir(args.out); figs=ensure_dir(out/"figures"); models_dir=ensure_dir(out/"models")
    panel=read_csv(args.processed/"A"/"q1_mixture_loss.csv", required=["pair_id","split","model_scale_label"])
    p_cols, loss_cols=detect_columns(panel)
    for c in p_cols+loss_cols: panel[c]=pd.to_numeric(panel[c], errors="coerce")
    panel=panel.dropna(subset=p_cols+loss_cols).copy()
    train=panel[panel["pair_id"]=="A4_A5"].copy()
    if len(train)==0: raise ValueError("A4_A5 training rows not found in q1_mixture_loss.csv")

    models, coef_df, feature_names, pairs=fit_models(train,p_cols,loss_cols,args.seed)
    dump_joblib(models, models_dir/"q1_mixture_models.joblib")
    write_csv(coef_df,out/"q1_mixture_coefficients.csv")
    # Final selected ElasticNet diagnostics. CV may emit warnings for discarded
    # candidate hyperparameters; these rows describe the actually selected model.
    diag_rows=[]
    for target,wrapped in models["scheffe_en"].items():
        m=wrapped.model
        n_iter=int(np.max(np.atleast_1d(getattr(m,"n_iter_",np.nan)))) if np.all(np.isfinite(np.atleast_1d(getattr(m,"n_iter_",np.nan)))) else np.nan
        diag_rows.append({
            "target":target,
            "alpha_selected":float(m.alpha_),
            "l1_ratio_selected":float(m.l1_ratio_),
            "n_iter":n_iter,
            "dual_gap":float(np.max(np.atleast_1d(getattr(m,"dual_gap_",np.nan)))),
            "max_iter_config":10000,
            "tol_config":1e-6,
            "hit_iteration_limit":bool(np.isfinite(n_iter) and n_iter>=10000),
        })
    write_csv(pd.DataFrame(diag_rows),out/"q1_elasticnet_diagnostics.csv")
    metrics=evaluate_across_splits(panel,models,p_cols,loss_cols)
    if not args.skip_gp:
        gp=fit_ilr_gp(train,p_cols,args.seed); dump_joblib(gp,models_dir/"q1_ilr_gp.joblib")
        gp_metrics=evaluate_gp(panel,gp,p_cols,loss_cols); metrics=pd.concat([metrics,gp_metrics],ignore_index=True,sort=False)
    write_csv(metrics,out/"q1_mixture_model_comparison.csv")

    inter=interaction_matrix(coef_df,p_cols,"scheffe_en")
    write_csv(inter.reset_index().rename(columns={"index":"domain"}),out/"q1_domain_interaction_matrix.csv")
    long=[]
    for i in range(len(p_cols)):
        for j in range(i+1,len(p_cols)):
            long.append({"domain_i":p_cols[i].removeprefix("p_"),"domain_j":p_cols[j].removeprefix("p_"),"gamma_mean":inter.iloc[i,j],
                         "relation":"complementary" if inter.iloc[i,j]<0 else "substitutive" if inter.iloc[i,j]>0 else "neutral"})
    interactions=pd.DataFrame(long).sort_values("gamma_mean")
    write_csv(interactions,out/"q1_domain_interactions_long.csv")

    # Q3 needs one fixed-p reference.  Do NOT continuously optimize the fitted
    # quadratic surface over the entire simplex: that can leave the empirical
    # support of A4/A5 and previously produced a non-physical negative Loss.
    # Instead select the best actually observed A4/A5 recipe by macro Loss.
    main_model=models["scheffe_en"]["loss_macro"]
    pstar, observed_star, predicted_star, res, ref_meta = select_empirical_reference_mixture(
        train,p_cols,loss_cols,model=main_model,method="scheffe_en"
    )
    opt=pd.DataFrame({"domain":[c.removeprefix("p_") for c in p_cols],"p_opt":pstar})
    opt["selection_method"]=ref_meta["selection_method"]
    opt["source_index"]=ref_meta["source_index"]
    opt["observed_macro_loss_1M"]=observed_star
    opt["predicted_macro_loss_1M"]=predicted_star
    # Canonical filename for the corrected fixed-p reference.
    write_csv(opt,out/"q1_reference_mixture.csv")
    write_json({**ref_meta,"selection_success":bool(res.success),"selection_message":str(res.message)},out/"q1_reference_mixture_summary.json")
    # Backward-compatible aliases so older Q3 commands do not break after Q1 rerun.
    write_csv(opt,out/"q1_optimal_mixture.csv")
    write_json({**ref_meta,"selection_success":bool(res.success),"selection_message":str(res.message)},out/"q1_optimal_mixture_summary.json")

    # Predictions for paper/use downstream.
    pred_rows=[]
    for method in ["linear","scheffe","scheffe_en"]:
        pred=predict_method(models,method,panel,p_cols,loss_cols)
        tmp=panel[["index","pair_id","split","model_scale_label"]].copy()
        tmp["method"]=method
        tmp["observed_loss_macro"]=panel[loss_cols].mean(axis=1)
        tmp["predicted_loss_macro"]=pred["loss_macro"]
        pred_rows.append(tmp)
    write_csv(pd.concat(pred_rows,ignore_index=True),out/"q1_mixture_predictions.csv")

    # Figures
    macro=metrics[metrics["target"]=="loss_macro"].copy()
    test1=macro[macro["pair_id"]=="A6_A7"]
    fig,axes,_=auto_break_barplot(data=test1,x="method",y="rmse",color="#4C72B0",figsize=(8,5),title="1M hold-out RMSE: mixture models",rotation=20); savefig(figs/"q1m_01_model_rmse.png",fig)
    fig,axes,_=auto_break_barplot(data=macro,x="model_scale_label",y="spearman",hue="method",figsize=(10,5),title="Cross-scale ranking stability"); savefig(figs/"q1m_02_cross_scale_spearman.png",fig)
    fig,ax=plt.subplots(figsize=(11,9)); sns.heatmap(inter,cmap="vlag",center=0,ax=ax); ax.set_title("Mean Scheffe-ElasticNet interaction coefficients"); savefig(figs/"q1m_03_interaction_heatmap.png",fig)
    top=pd.concat([interactions.head(12),interactions.tail(12)]).copy(); top["pair"]=top["domain_i"]+" × "+top["domain_j"]
    fig,axes,_=auto_break_barplot(data=top,x="gamma_mean",y="pair",color="#4C72B0",figsize=(9,8),title="Strongest complementary / substitutive domain pairs",zero_line=True); savefig(figs/"q1m_04_top_interactions.png",fig)
    pr=pd.concat(pred_rows,ignore_index=True); pr=pr[(pr.pair_id=="A6_A7") & (pr.method=="scheffe_en")]
    fig,ax=plt.subplots(figsize=(6,6)); sns.scatterplot(data=pr,x="observed_loss_macro",y="predicted_loss_macro",ax=ax); mn=min(pr.observed_loss_macro.min(),pr.predicted_loss_macro.min()); mx=max(pr.observed_loss_macro.max(),pr.predicted_loss_macro.max()); ax.plot([mn,mx],[mn,mx],ls="--"); ax.set_title("Scheffe-ElasticNet: observed vs predicted (1M test)"); savefig(figs/"q1m_05_observed_predicted.png",fig)
    fig,axes,_=auto_break_barplot(data=opt.sort_values("p_opt"),x="p_opt",y="domain",color="#4C72B0",figsize=(9,6),title="Empirical fixed-p reference: best observed A4/A5 recipe"); savefig(figs/"q1m_06_optimal_mixture.png",fig)

    print(f"Q1 mixture complete: {out}; fixed-p reference observed macro loss={observed_star:.6f}; model prediction at reference={predicted_star:.6f}")

if __name__=="__main__": main()
