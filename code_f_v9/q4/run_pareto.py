from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from common.io_utils import ensure_dir,read_csv,write_csv
from common.plotting import setup_style,savefig
from q4.capability import BENCH_COLS,pareto_flags


def main():
    ap=argparse.ArgumentParser(description="Q4 six-dimensional Benchmark Pareto frontier")
    ap.add_argument("--processed",type=Path,required=True); ap.add_argument("--capability",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("results/q4_pareto"))
    a=ap.parse_args(); setup_style(); out=ensure_dir(a.out); figs=ensure_dir(out/"figures")
    lb=read_csv(a.processed/"C"/"q4_leaderboard.csv",required=["model","model_key",*BENCH_COLS]);cap=read_csv(a.capability/"q4_capability_scores.csv")
    df=lb.merge(cap[["model_key","cap_pca","pc2"]],on="model_key",how="inner").dropna(subset=BENCH_COLS);df["submission_date"]=pd.to_datetime(df.submission_date,errors="coerce")
    df["pareto_flag"]=pareto_flags(df[BENCH_COLS].to_numpy(float),show_progress=True,desc="Q4 global Pareto")
    rows=[]
    yearly_groups=list(df.dropna(subset=["submission_date"]).assign(year=lambda x:x.submission_date.dt.year).groupby("year"))
    for y,g in tqdm(yearly_groups,desc="Q4 yearly Pareto",unit="year"):
        flags=pareto_flags(g[BENCH_COLS].to_numpy(float),show_progress=False)
        for idx,flag in zip(g.index,flags): rows.append({"model_key":g.loc[idx,"model_key"],"pareto_year":int(y),"pareto_year_flag":bool(flag)})
    yearly=pd.DataFrame(rows); write_csv(df,out/"q4_pareto_global.csv"); write_csv(yearly,out/"q4_pareto_yearly.csv")
    fig,ax=plt.subplots(figsize=(8,6));sns.scatterplot(data=df,x="cap_pca",y="pc2",hue="pareto_flag",style="pareto_flag",alpha=.65,ax=ax);ax.set_title("Six-dimensional Benchmark Pareto frontier in PCA space");savefig(figs/"q4p_01_pareto_pca.png",fig)
    tmp=df.dropna(subset=["submission_date"]).copy();tmp["year"]=tmp.submission_date.dt.year;cnt=tmp.groupby("year").pareto_flag.agg(["sum","count"]).reset_index();cnt["pareto_share"]=cnt["sum"]/cnt["count"]
    fig,ax=plt.subplots(figsize=(9,5));sns.lineplot(data=cnt,x="year",y="pareto_share",marker="o",ax=ax);ax.set_title("Share of models on the Pareto frontier by year");savefig(figs/"q4p_02_pareto_share_time.png",fig)
    print(f"Q4 Pareto analysis complete: {out}; global Pareto models={int(df.pareto_flag.sum())}")
if __name__=="__main__":main()
