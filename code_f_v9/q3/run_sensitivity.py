from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from common.io_utils import ensure_dir,read_csv,read_json,write_csv
from common.plotting import setup_style,savefig
from q3.optimizer import PredictorEnsemble,BoundsSpec,solve_hybrid,result_row
from q3.run_optimization import derive_bounds,load_fixed_p


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--processed",type=Path,required=True); ap.add_argument("--q1-quality",type=Path,required=True); ap.add_argument("--q1-mixture",type=Path,required=True); ap.add_argument("--q2",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("results/q3")); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--fast",action="store_true")
    a=ap.parse_args(); setup_style(); out=ensure_dir(a.out); figs=ensure_dir(out/"figures")
    Q0=float(read_json(a.q1_quality/"q1_quality_summary.json")["Q0_global"]); spec=derive_bounds(a.processed,Q0); ens=PredictorEnsemble(a.q2,(a.q2/"bootstrap"/"q2_bootstrap_scaling_params.csv") if (a.q2/"bootstrap"/"q2_bootstrap_scaling_params.csv").exists() else None,max_samples=40 if a.fast else 200)
    ctx=read_csv(a.processed/"C"/"q3_context.csv"); contexts=np.sort(pd.to_numeric(ctx.max_position_embeddings,errors="coerce").dropna().unique()); contexts=contexts[contexts>0]
    # keep actual unique values, but if dozens of per-model values, use empirical quantile representatives.
    if len(contexts)>10: contexts=np.unique(np.quantile(contexts,[0,.25,.5,.75,1]).round())
    rows=[]
    jobs=[(kind,L) for kind in ["exponential","power","log"] for L in contexts]
    for kind,L in tqdm(jobs, desc="Q3 cost/context sensitivity", unit="case"):
        r,t=solve_hybrid(ens,1e22,Q0,float(L),kind,spec,True,None,"mean_cvar",.2,a.seed,fast=a.fast,show_progress=True)
        rows.append(result_row(r,t,ens,1e22,Q0,float(L),kind,True,None,"mean_cvar",.2,"DE+SLSQP"))
    df=pd.DataFrame(rows); write_csv(df,out/"q3_cost_context_sensitivity.csv")
    fig,ax=plt.subplots(figsize=(9,5)); sns.lineplot(data=df,x="L_ctx",y="Q_opt",hue="cost_kind",marker="o",ax=ax); ax.axvline(30000,ls="--",lw=1,label="Lcrit=30000"); ax.set_xscale("log"); ax.set_title("Quality-cost and context sensitivity"); savefig(figs/"q3_06_context_quality_sensitivity.png",fig)
    ratio=pd.DataFrame({"L_ctx":contexts,"Cattn_over_Ctrain":2e-4*contexts/6}); fig,ax=plt.subplots(figsize=(8,5)); sns.lineplot(data=ratio,x="L_ctx",y="Cattn_over_Ctrain",marker="o",ax=ax); ax.axhline(1,ls="--"); ax.axvline(30000,ls="--"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title("Attention/training cost ratio; analytic critical context=30000"); savefig(figs/"q3_07_context_critical.png",fig)
    print(f"Q3 sensitivity complete: {out}")
if __name__=="__main__":main()
