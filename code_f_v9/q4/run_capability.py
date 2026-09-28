from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.io_utils import ensure_dir,read_csv,write_csv,dump_joblib
from common.plotting import setup_style,savefig,auto_break_barplot
from common.random_utils import set_seed
from q4.capability import BENCH_COLS,fit_capability,capability_stability,aggregate_c8


def main():
    ap=argparse.ArgumentParser(description="Q4 capability index comparison + mandatory C8 task aggregation")
    ap.add_argument("--processed",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("results/q4_capability")); ap.add_argument("--bootstrap",type=int,default=1000); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--fast",action="store_true")
    a=ap.parse_args(); set_seed(a.seed); setup_style(); out=ensure_dir(a.out); figs=ensure_dir(out/"figures"); models=ensure_dir(out/"models")
    lb=read_csv(a.processed/"C"/"q4_leaderboard.csv",required=["model","model_key",*BENCH_COLS]); lb["submission_date"]=pd.to_datetime(lb.get("submission_date"),errors="coerce")
    for c in BENCH_COLS:lb[c]=pd.to_numeric(lb[c],errors="coerce")
    bundle,scores,load,ev=fit_capability(lb); dump_joblib(bundle,models/"capability_pca.joblib"); write_csv(scores,out/"q4_capability_scores.csv"); write_csv(load,out/"q4_pca_loadings.csv"); write_csv(ev,out/"q4_pca_explained_variance.csv")
    nboot=min(a.bootstrap,30) if a.fast else a.bootstrap; bst,leave=capability_stability(lb,bundle,nboot,a.seed); write_csv(bst,out/"q4_capability_bootstrap_stability.csv"); write_csv(leave,out/"q4_capability_leave_one_benchmark.csv")
    # Three methods scorecard: no external ground truth; report rank agreement and smoothness proxies.
    comp=[]
    bootmap={"cap_raw_mean":"raw_mean_rank_spearman","cap_z_mean":"zmean_rank_spearman","cap_pca":"pca_rank_spearman"}
    tmpyear=scores.dropna(subset=["submission_date"]).copy(); tmpyear["year"]=tmpyear.submission_date.dt.year
    for col in ["cap_raw_mean","cap_z_mean","cap_pca"]:
        rho=float(scores[[col,"cap_raw_mean"]].corr(method="spearman").iloc[0,1])
        annual=tmpyear.groupby("year")[col].median().sort_index().to_numpy(float); rough=float(np.mean(np.diff(annual,n=2)**2)) if len(annual)>=3 else np.nan
        comp.append({"method":col,"spearman_vs_raw_mean":rho,"score_std":float(scores[col].std()),"bootstrap_rank_spearman_mean":float(bst[bootmap[col]].mean()),"annual_trend_roughness":rough})
    write_csv(pd.DataFrame(comp),out/"q4_capability_method_comparison.csv")

    # C3 is mandatory/explicit time-series evidence. Apply the PCA fitted on C2 to the six C3 benchmarks.
    ts_path=a.processed/"C"/"q4_timeseries.csv"
    if ts_path.exists():
        ts=read_csv(ts_path,required=["model","model_key","year",*BENCH_COLS])
        for c in BENCH_COLS: ts[c]=pd.to_numeric(ts[c],errors="coerce")
        good=ts.dropna(subset=BENCH_COLS).copy()
        if len(good):
            good["cap_pca"]=bundle.transform(good); good["cap_raw_mean"]=good[BENCH_COLS].mean(axis=1)
            write_csv(good,out/"q4_timeseries_capability.csv")
            annual=good.groupby("year",as_index=False).agg(median_capability=("cap_pca","median"),q95_capability=("cap_pca",lambda x:x.quantile(.95)),n_models=("model","count"))
            write_csv(annual,out/"q4_timeseries_annual_summary.csv")

    tasks=read_csv(a.processed/"C"/"q4_task_metrics.csv",required=["model_key","model_name","task_group","task","metric","value"])
    for c in ["value","n_samples_effective"]:
        if c in tasks:tasks[c]=pd.to_numeric(tasks[c],errors="coerce")
    task,group,spec=aggregate_c8(tasks); write_csv(task,out/"q4_c8_task_scores.csv"); write_csv(group,out/"q4_c8_group_scores.csv"); write_csv(spec,out/"q4_c8_specialization.csv")
    scored=scores.merge(spec,on="model_key",how="left",suffixes=("","_c8")); write_csv(scored,out/"q4_capability_with_c8.csv")

    # Figures
    complete=lb.dropna(subset=BENCH_COLS)
    fig,ax=plt.subplots(figsize=(7,6)); sns.heatmap(complete[BENCH_COLS].corr(method="spearman"),annot=True,fmt=".2f",cmap="vlag",center=0,ax=ax); ax.set_title("Benchmark Spearman correlation"); savefig(figs/"q4_01_benchmark_correlation.png",fig)
    fig,axes,_=auto_break_barplot(data=ev,x="component",y="explained_variance_ratio",color="#4C72B0",figsize=(7,5),title="PCA scree plot"); savefig(figs/"q4_02_pca_scree.png",fig)
    fig,axes,_=auto_break_barplot(data=load.sort_values("pc1_loading"),x="pc1_loading",y="benchmark",color="#4C72B0",figsize=(8,5),title="PC1 benchmark loadings"); savefig(figs/"q4_03_pca_loadings.png",fig)
    fig,ax=plt.subplots(figsize=(8,6)); sns.scatterplot(data=scores,x="cap_pca",y="pc2",hue="model_type_bucket" if "model_type_bucket" in scores else None,alpha=.55,s=25,ax=ax); ax.set_title("Capability map: PC1 vs PC2"); savefig(figs/"q4_04_pca_map.png",fig)
    if "submission_date" in scores:
        ts=scores.dropna(subset=["submission_date"]).copy(); ts["year_month"]=ts["submission_date"].dt.to_period("M").dt.to_timestamp(); trend=ts.groupby("year_month",as_index=False).cap_pca.median()
        fig,ax=plt.subplots(figsize=(10,5)); sns.scatterplot(data=ts,x="submission_date",y="cap_pca",hue="model_type_bucket" if "model_type_bucket" in ts else None,alpha=.35,s=18,ax=ax); sns.lineplot(data=trend,x="year_month",y="cap_pca",linewidth=2,ax=ax,label="Monthly median"); ax.set_title("Capability evolution over submission time"); savefig(figs/"q4_05_capability_time.png",fig)
    if "model_type_bucket" in scores:
        fig,ax=plt.subplots(figsize=(9,5)); sns.violinplot(data=scores,x="model_type_bucket",y="cap_pca",cut=0,inner="quartile",ax=ax); ax.tick_params(axis="x",rotation=25); ax.set_title("Capability distribution by model type"); savefig(figs/"q4_06_capability_type_violin.png",fig)
    topkeys=scores.nlargest(30,"cap_pca").model_key; heat=group[group.model_key.isin(topkeys)].pivot_table(index="model_name",columns="task_group",values="group_score",aggfunc="mean")
    if len(heat):
        fig,ax=plt.subplots(figsize=(9,max(6,.25*len(heat)))); sns.heatmap(heat,cmap="viridis",ax=ax); ax.set_title("C8 detailed-task aggregate scores for top capability models"); savefig(figs/"q4_07_c8_task_heatmap.png",fig)
    ss=scored.dropna(subset=["c8_specialization_sd"])
    if len(ss):
        fig,ax=plt.subplots(figsize=(8,6)); sns.scatterplot(data=ss,x="cap_pca",y="c8_specialization_sd",hue="model_type_bucket" if "model_type_bucket" in ss else None,alpha=.6,ax=ax); ax.set_title("Overall capability vs specialization imbalance"); savefig(figs/"q4_08_capability_specialization.png",fig)
    annual_path=out/"q4_timeseries_annual_summary.csv"
    if annual_path.exists():
        annual=pd.read_csv(annual_path); fig,ax=plt.subplots(figsize=(9,5)); sns.lineplot(data=annual,x="year",y="median_capability",marker="o",label="Median",ax=ax); sns.lineplot(data=annual,x="year",y="q95_capability",marker="o",label="95th percentile",ax=ax); ax.set_title("C3 historical capability trajectory under the C2-fitted PCA scale"); savefig(figs/"q4_09_C3_timeseries_capability.png",fig)
    print(f"Q4 capability + C8 complete: {out}")
if __name__=="__main__":main()
