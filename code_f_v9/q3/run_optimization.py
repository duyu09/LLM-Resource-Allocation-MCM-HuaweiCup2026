from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from common.io_utils import ensure_dir, read_csv, read_json, write_csv
from common.plotting import setup_style, savefig, auto_break_barplot
from common.random_utils import set_seed
from q3.optimizer import PredictorEnsemble, BoundsSpec, solve_slsqp_multistart, solve_de, solve_hybrid, result_row, finite_difference_kkt


def derive_bounds(processed,Q0):
    s=pd.read_csv(processed/"B"/"q2_scaling.csv"); n=s.N_params_B.dropna(); d=s.D_tokens_B.dropna(); n=n[n>0]; d=d[d>0]
    lp=processed/"B"/"q2_large_models.csv"
    if lp.exists():
        lg=pd.read_csv(lp)
        ln=pd.to_numeric(lg.get("N_params_B"),errors="coerce").dropna() if "N_params_B" in lg else pd.Series(dtype=float)
        ld=pd.to_numeric(lg.get("D_tokens_B"),errors="coerce").dropna() if "D_tokens_B" in lg else pd.Series(dtype=float)
        nmax=max(float(n.max()),float(ln.max()) if len(ln) else float(n.max()))
        dmax=max(float(d.max()),float(ld.max()) if len(ld) else float(d.max()))
    else:nmax=float(n.max()); dmax=float(d.max())
    logN=(float(np.log10(max(n.min()*1e9*.2,1e5))),float(np.log10(min(max(nmax*1e9*1.5,1e9),1e14))))
    logD=(float(np.log10(max(d.min()*1e9*.2,1e6))),float(np.log10(min(max(dmax*1e9*2,1e10),1e17))))
    return BoundsSpec(logN,logD,(Q0,1.0))


def load_fixed_p(q1_dir,return_domains=False):
    q1_dir=Path(q1_dir)
    preferred=q1_dir/"q1_reference_mixture.csv"
    legacy=q1_dir/"q1_optimal_mixture.csv"
    path=preferred if preferred.exists() else legacy
    if not path.exists():
        raise FileNotFoundError(
            f"Q3 fixed-p requires {preferred.name}. Rerun q1.run_mixture with the corrected code."
        )
    x=pd.read_csv(path)
    if "p_opt" not in x.columns:
        raise ValueError(f"{path} is missing p_opt")
    p=pd.to_numeric(x.p_opt,errors="coerce").to_numpy(float)
    if len(p)!=17 or not np.all(np.isfinite(p)) or np.any(p<0) or p.sum()<=0:
        raise ValueError(f"Invalid fixed-p composition in {path}")
    p=p/p.sum()

    # Reject the known legacy failure mode: an off-support continuous optimum
    # whose fitted macro cross-entropy is negative.  This prevents stale v8 Q1
    # outputs from silently contaminating Q3 after the code has been upgraded.
    if "selection_method" in x.columns:
        methods=x["selection_method"].dropna().astype(str).unique().tolist()
        if methods and methods[0]!="best_observed_A4_A5_macro_loss":
            raise ValueError(f"Unsupported fixed-p selection method in {path}: {methods[0]}")
    elif path==legacy:
        summary=q1_dir/"q1_optimal_mixture_summary.json"
        if summary.exists():
            meta=read_json(summary)
            pred=meta.get("predicted_macro_loss_1M",None)
            if pred is not None and np.isfinite(float(pred)) and float(pred)<0:
                raise ValueError(
                    "Legacy q1_optimal_mixture.csv has a negative predicted cross-entropy. "
                    "Rerun q1.run_mixture; the corrected pipeline uses an observed-support reference recipe."
                )
        raise ValueError(
            "Legacy q1_optimal_mixture.csv lacks corrected selection metadata. "
            "Rerun q1.run_mixture before Q3 so fixed-p is selected from observed A4/A5 support."
        )

    domains=x["domain"].astype(str).tolist() if "domain" in x.columns else [f"p_{i:02d}" for i in range(len(p))]
    return (p,domains) if return_domains else p



def _plot_q3_01_stacked(core: pd.DataFrame, figs: Path) -> None:
    """Draw q3_01 as one 100% stacked bar per compute budget.

    Uses the robust-joint policy when it is available.  The three cost shares are
    renormalized by their row sum before plotting so every bar has height exactly
    one; this makes the figure a composition chart rather than three side-by-side
    bars.  Raw, unnormalized shares remain unchanged in the CSV outputs.
    """
    required={"budget","algorithm","r_train","r_quality","r_attn"}
    missing=sorted(required.difference(core.columns))
    if missing:
        raise ValueError("q3_core_budget_strategies.csv is missing required columns: "+", ".join(missing))
    df=core.copy()
    for c in ["budget","r_train","r_quality","r_attn"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    if (df["algorithm"]=="DE+SLSQP_robust_joint").any():
        df=df[df["algorithm"]=="DE+SLSQP_robust_joint"].copy()
    else:
        # Fallback for older result files: one row per budget, preferring the last
        # available strategy in file order rather than failing the replot command.
        df=df.dropna(subset=["budget"]).sort_values("budget").groupby("budget",as_index=False).tail(1)
    df=df.dropna(subset=["budget","r_train","r_quality","r_attn"]).sort_values("budget")
    if df.empty:
        raise ValueError("No valid rows are available to draw q3_01.")
    shares=df[["r_train","r_quality","r_attn"]].clip(lower=0).to_numpy(float)
    denom=shares.sum(axis=1,keepdims=True)
    if np.any(denom<=0):
        raise ValueError("At least one budget has non-positive total cost share.")
    shares=shares/denom
    labels=[f"1e{int(round(np.log10(x)))}" for x in df["budget"].to_numpy(float)]
    palette=sns.color_palette("Set2",3)
    fig,ax=plt.subplots(figsize=(8,5))
    bottom=np.zeros(len(df),dtype=float)
    names=["Training","Quality","Attention"]
    for j,name in enumerate(names):
        vals=shares[:,j]
        bars=ax.bar(labels,vals,bottom=bottom,label=name,color=palette[j],width=.65)
        for i,(b,v) in enumerate(zip(bars,vals)):
            if v>=0.035:
                ax.text(b.get_x()+b.get_width()/2,bottom[i]+v/2,f"{100*v:.1f}%",ha="center",va="center",fontsize=9)
        bottom+=vals
    ax.set_ylim(0,1)
    ax.set_xlabel("Compute budget (FLOPs)")
    ax.set_ylabel("Normalized cost share")
    ax.set_title("Optimal compute allocation by budget")
    ax.legend(title="Cost component",bbox_to_anchor=(1.02,1),loc="upper left",borderaxespad=0)
    ax.grid(axis="x",visible=False)
    savefig(figs/"q3_01_budget_cost_shares.png",fig)


def _refresh_fixed_p_only(args,out,figs):
    required=[("--processed",args.processed),("--q1-quality",args.q1_quality),("--q1-mixture",args.q1_mixture),("--q2",args.q2)]
    missing=[name for name,val in required if val is None]
    if missing:
        raise ValueError("--refresh-fixed-p requires "+", ".join(missing))
    core_csv=out/"q3_core_budget_strategies.csv"
    if not core_csv.exists():
        raise FileNotFoundError(f"--refresh-fixed-p requires existing {core_csv}; run full Q3 once first")

    core=pd.read_csv(core_csv)
    qsum=read_json(args.q1_quality/"q1_quality_summary.json")
    Q0=float(qsum["Q0_global"])
    pfix,domains=load_fixed_p(args.q1_mixture,return_domains=True)
    write_csv(pd.DataFrame({"p_index":[f"p_{i:02d}" for i in range(len(domains))],"domain":domains}),out/"q3_domain_order.csv")
    ctx=read_csv(args.processed/"C"/"q3_context.csv",required=["max_position_embeddings"])
    vals=np.sort(ctx.max_position_embeddings.dropna().astype(float).unique()); vals=vals[vals>0]
    Lctx=float(args.context if args.context is not None else np.median(vals))
    boot=args.bootstrap_csv or (args.q2/"bootstrap"/"q2_bootstrap_scaling_params.csv")
    ens=PredictorEnsemble(args.q2,boot if Path(boot).exists() else None,max_samples=40 if args.fast else 300)
    spec=derive_bounds(args.processed,Q0)

    new_rows=[]
    for C in tqdm([1e19,1e22,1e24],desc="Q3 refresh fixed-p",unit="budget"):
        old=core[(pd.to_numeric(core["budget"],errors="coerce")==C)&core["algorithm"].eq("DE+SLSQP_fixed_p")]
        x0=None
        if len(old):
            rr=old.iloc[0]
            try:
                x0=np.array([np.log10(float(rr.N_opt)),np.log10(float(rr.D_opt)),float(rr.Q_opt)],float)
            except Exception:
                x0=None
        r,t=solve_hybrid(ens,C,Q0,Lctx,args.cost_kind,spec,False,pfix,"mean",0,args.seed,x0=x0,fast=args.fast,show_progress=True)
        new_rows.append(result_row(r,t,ens,C,Q0,Lctx,args.cost_kind,False,pfix,"mean",0,"DE+SLSQP_fixed_p"))

    keep=core[~core["algorithm"].eq("DE+SLSQP_fixed_p")].copy()
    core=pd.concat([keep,pd.DataFrame(new_rows)],ignore_index=True,sort=False)
    core=core.sort_values(["budget","algorithm"]).reset_index(drop=True)
    write_csv(core,core_csv)

    # Recompute the only derived table that directly contrasts fixed-p with joint-p.
    value=[]
    for C in [1e19,1e22,1e24]:
        f=core[(core.budget==C)&core.algorithm.eq("DE+SLSQP_fixed_p")].iloc[0]
        j=core[(core.budget==C)&core.algorithm.eq("DE+SLSQP_joint")].iloc[0]
        value.append({"budget":C,"loss_fixed_p":f.expected_loss,"loss_joint_p":j.expected_loss,"value_adaptive_p":(f.expected_loss-j.expected_loss)/(abs(f.expected_loss)+1e-12)})
    write_csv(pd.DataFrame(value),out/"q3_value_of_adaptive_mixture.csv")

    # Figures depending on the core-strategy table. q3_01 uses robust-joint only,
    # but is regenerated for file consistency; q3_02/q3_03 include fixed-p.
    _plot_q3_01_stacked(core,figs)
    qlong=core.melt(id_vars=["budget","algorithm"],value_vars=["N_opt","D_opt"],var_name="resource",value_name="value")
    fig,ax=plt.subplots(figsize=(8,5)); sns.lineplot(data=qlong,x="budget",y="value",hue="resource",style="algorithm",markers=True,ax=ax); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title("Optimal N and D across budgets"); savefig(figs/"q3_02_optimal_N_D.png",fig)
    fig,ax=plt.subplots(figsize=(8,5)); sns.lineplot(data=core,x="budget",y="Q_opt",hue="algorithm",marker="o",ax=ax); ax.set_xscale("log"); ax.set_title("Optimal quality Q across budgets"); savefig(figs/"q3_03_optimal_Q.png",fig)
    print(f"Q3 fixed-p refresh complete: {out}; Q0={Q0:.4f}; Lctx={Lctx:g}")


def main():
    ap=argparse.ArgumentParser(description="Q3 core optimization + strategy/optimizer/risk comparisons.")
    # These inputs are required for optimization, but not for --replot-q3-01.
    ap.add_argument("--processed",type=Path,required=False); ap.add_argument("--q1-quality",type=Path,required=False); ap.add_argument("--q1-mixture",type=Path,required=False); ap.add_argument("--q2",type=Path,required=False)
    ap.add_argument("--out",type=Path,default=Path("results/q3")); ap.add_argument("--bootstrap-csv",type=Path,default=None)
    ap.add_argument("--context",type=float,default=None); ap.add_argument("--cost-kind",choices=["exponential","power","log"],default="exponential")
    ap.add_argument("--risk-lambda",type=float,default=.2); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--fast",action="store_true")
    ap.add_argument("--optimizer-repeats",type=int,default=5,help="Repeated random seeds for E5 optimizer stability comparison.")
    ap.add_argument("--replot-q3-01",action="store_true",help="Rebuild q3_01_budget_cost_shares.png from an existing q3_core_budget_strategies.csv in --out, then exit without optimization.")
    ap.add_argument("--refresh-fixed-p",action="store_true",help="Recompute only the three fixed-p rows after Q1 reference-mixture correction; preserve existing joint/robust rows and unrelated Q3 outputs.")
    ap.add_argument("--slsqp-starts",type=int,default=100,help="Random starts for the multi-start SLSQP baseline.")
    args=ap.parse_args(); set_seed(args.seed); setup_style(); out=ensure_dir(args.out); figs=ensure_dir(out/"figures")
    if args.replot_q3_01:
        core_csv=out/"q3_core_budget_strategies.csv"
        if not core_csv.exists():
            raise FileNotFoundError(f"--replot-q3-01 requires an existing file: {core_csv}")
        _plot_q3_01_stacked(pd.read_csv(core_csv),figs)
        print(f"q3_01 regenerated from existing results: {figs/'q3_01_budget_cost_shares.png'}")
        return
    if args.refresh_fixed_p:
        _refresh_fixed_p_only(args,out,figs)
        return
    missing_args=[name for name,val in [("--processed",args.processed),("--q1-quality",args.q1_quality),("--q1-mixture",args.q1_mixture),("--q2",args.q2)] if val is None]
    if missing_args:
        ap.error("the following arguments are required unless --replot-q3-01 is used: "+", ".join(missing_args))
    qsum=read_json(args.q1_quality/"q1_quality_summary.json"); Q0=float(qsum["Q0_global"]); pfix,domains=load_fixed_p(args.q1_mixture,return_domains=True); write_csv(pd.DataFrame({"p_index":[f"p_{i:02d}" for i in range(len(domains))],"domain":domains}),out/"q3_domain_order.csv")
    ctx=read_csv(args.processed/"C"/"q3_context.csv",required=["max_position_embeddings"]); vals=np.sort(ctx.max_position_embeddings.dropna().astype(float).unique()); vals=vals[vals>0]
    Lctx=float(args.context if args.context is not None else np.median(vals))
    boot=args.bootstrap_csv or (args.q2/"bootstrap"/"q2_bootstrap_scaling_params.csv")
    ens=PredictorEnsemble(args.q2,boot if Path(boot).exists() else None,max_samples=40 if args.fast else 300)
    spec=derive_bounds(args.processed,Q0)

    rows=[]
    budgets=[1e19,1e22,1e24]
    budget_bar=tqdm(budgets, desc="Q3 core budgets", unit="budget")
    for C in budget_bar:
        label=f"1e{int(np.log10(C))}"
        stage=tqdm(total=3,desc=f"Q3 {label} strategies",unit="strategy",leave=False)
        # Fixed p, nominal
        budget_bar.set_postfix(budget=label,stage="fixed-p nominal")
        stage.set_postfix_str("fixed-p nominal")
        r,t=solve_hybrid(ens,C,Q0,Lctx,args.cost_kind,spec,False,pfix,"mean",0,args.seed,fast=args.fast,show_progress=True)
        rows.append(result_row(r,t,ens,C,Q0,Lctx,args.cost_kind,False,pfix,"mean",0,"DE+SLSQP_fixed_p")); stage.update(1)
        # Joint nominal
        budget_bar.set_postfix(budget=label,stage="joint nominal")
        stage.set_postfix_str("joint nominal")
        r,t=solve_hybrid(ens,C,Q0,Lctx,args.cost_kind,spec,True,None,"mean",0,args.seed+1,fast=args.fast,show_progress=True)
        rows.append(result_row(r,t,ens,C,Q0,Lctx,args.cost_kind,True,None,"mean",0,"DE+SLSQP_joint")); stage.update(1)
        # Joint robust main
        budget_bar.set_postfix(budget=label,stage="robust joint")
        stage.set_postfix_str("robust joint")
        r,t=solve_hybrid(ens,C,Q0,Lctx,args.cost_kind,spec,True,None,"mean_cvar",args.risk_lambda,args.seed+2,fast=args.fast,show_progress=True)
        rr=result_row(r,t,ens,C,Q0,Lctx,args.cost_kind,True,None,"mean_cvar",args.risk_lambda,"DE+SLSQP_robust_joint"); rr.update(finite_difference_kkt(ens,rr,Q0,args.cost_kind)); rows.append(rr); stage.update(1); stage.close()
    core=pd.DataFrame(rows); write_csv(core,out/"q3_core_budget_strategies.csv")

    # E5 optimizer comparison at middle budget, joint nominal.
    # Repeat random seeds so the paper can report feasibility, runtime and objective stability.
    C=1e22; orows=[]
    nrep=max(2,min(args.optimizer_repeats,2)) if args.fast else max(1,args.optimizer_repeats)
    for rep in tqdm(range(nrep), desc="Q3 optimizer repeats", unit="repeat"):
        seed=args.seed+100*rep
        r,t=solve_slsqp_multistart(ens,C,Q0,Lctx,args.cost_kind,spec,True,None,"mean",0,
                                  n_starts=3 if args.fast else args.slsqp_starts,seed=seed,show_progress=True)
        if r is not None:
            rr=result_row(r,t,ens,C,Q0,Lctx,args.cost_kind,True,None,"mean",0,"multi_start_SLSQP")
            rr.update(finite_difference_kkt(ens,rr,Q0,args.cost_kind)); rr["repeat"]=rep; orows.append(rr)
        r,t=solve_de(ens,C,Q0,Lctx,args.cost_kind,spec,True,None,"mean",0,seed,
                     maxiter=2 if args.fast else 500,popsize=2 if args.fast else 20,show_progress=True)
        rr=result_row(r,t,ens,C,Q0,Lctx,args.cost_kind,True,None,"mean",0,"DE")
        rr.update(finite_difference_kkt(ens,rr,Q0,args.cost_kind)); rr["repeat"]=rep; orows.append(rr)
        r,t=solve_hybrid(ens,C,Q0,Lctx,args.cost_kind,spec,True,None,"mean",0,seed,fast=args.fast,show_progress=True)
        rr=result_row(r,t,ens,C,Q0,Lctx,args.cost_kind,True,None,"mean",0,"DE+SLSQP")
        rr.update(finite_difference_kkt(ens,rr,Q0,args.cost_kind)); rr["repeat"]=rep; orows.append(rr)
    optraw=pd.DataFrame(orows)
    optraw["feasible"]=(optraw.C_total<=optraw.budget*(1+1e-8)) & optraw.success.astype(bool)
    write_csv(optraw,out/"q3_optimizer_repeated_runs.csv")
    optcmp=(optraw.groupby("algorithm",as_index=False)
            .agg(n_runs=("repeat","count"), feasible_rate=("feasible","mean"),
                 best_objective=("objective","min"), mean_objective=("objective","mean"),
                 std_objective=("objective","std"), mean_runtime_sec=("runtime_sec","mean"),
                 mean_kkt_spread=("kkt_ratio_spread","mean"), max_kkt_spread=("kkt_ratio_spread","max")))
    write_csv(optcmp,out/"q3_optimizer_comparison.csv")

    # E6 risk objective comparison at middle budget.
    rrows=[]
    for mode,lam in tqdm([("mean",0),("mean_sd",args.risk_lambda),("mean_cvar",args.risk_lambda)], desc="Q3 risk objectives", unit="mode"):
        r,t=solve_hybrid(ens,C,Q0,Lctx,args.cost_kind,spec,True,None,mode,lam,args.seed+3,fast=args.fast,show_progress=True)
        rr=result_row(r,t,ens,C,Q0,Lctx,args.cost_kind,True,None,mode,lam,"DE+SLSQP")
        rrows.append(rr)
    risk=pd.DataFrame(rrows)
    # Scenario-wise regret: compare each risk policy against the best of the candidate policies
    # under every bootstrap model draw. This operationalizes Regret=L(x)-min_x' L(x').
    scenario_losses=[]
    for _,rr in risk.iterrows():
        p=np.array([rr[f"p_{i:02d}"] for i in range(17)],float)
        scenario_losses.append(ens.predict_samples(rr.N_opt,rr.D_opt,rr.Q_opt,p))
    if scenario_losses:
        Lmat=np.vstack(scenario_losses); oracle=Lmat.min(axis=0); regrets=Lmat-oracle
        for i in range(len(risk)):
            rg=regrets[i]; risk.loc[i,"mean_regret"]=float(np.mean(rg)); risk.loc[i,"p95_regret"]=float(np.quantile(rg,.95));
            k=max(1,int(np.ceil(.05*len(rg)))); risk.loc[i,"worst5pct_mean_regret"]=float(np.mean(np.sort(rg)[-k:]))
    write_csv(risk,out/"q3_risk_objective_comparison.csv")

    # Value of adaptive p.
    value=[]
    for C in budgets:
        f=core[(core.budget==C)&core.algorithm.eq("DE+SLSQP_fixed_p")].iloc[0]; j=core[(core.budget==C)&core.algorithm.eq("DE+SLSQP_joint")].iloc[0]
        value.append({"budget":C,"loss_fixed_p":f.expected_loss,"loss_joint_p":j.expected_loss,"value_adaptive_p":(f.expected_loss-j.expected_loss)/(abs(f.expected_loss)+1e-12)})
    write_csv(pd.DataFrame(value),out/"q3_value_of_adaptive_mixture.csv")

    # Figures
    _plot_q3_01_stacked(core,figs)
    qlong=core.melt(id_vars=["budget","algorithm"],value_vars=["N_opt","D_opt"],var_name="resource",value_name="value")
    fig,ax=plt.subplots(figsize=(8,5)); sns.lineplot(data=qlong,x="budget",y="value",hue="resource",style="algorithm",markers=True,ax=ax); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title("Optimal N and D across budgets"); savefig(figs/"q3_02_optimal_N_D.png",fig)
    fig,ax=plt.subplots(figsize=(8,5)); sns.lineplot(data=core,x="budget",y="Q_opt",hue="algorithm",marker="o",ax=ax); ax.set_xscale("log"); ax.set_title("Optimal quality Q across budgets"); savefig(figs/"q3_03_optimal_Q.png",fig)
    fig,axes,_=auto_break_barplot(data=optcmp,x="algorithm",y="best_objective",color="#4C72B0",figsize=(8,5),title="Optimizer comparison",rotation=20); savefig(figs/"q3_04_optimizer_comparison.png",fig)
    fig,axes,_=auto_break_barplot(data=risk,x="risk_mode",y="cvar95",color="#4C72B0",figsize=(8,5),title="Tail-risk comparison of optimization objectives"); savefig(figs/"q3_05_risk_comparison.png",fig)
    print(f"Q3 core optimization complete: {out}; Q0={Q0:.4f}; Lctx={Lctx:g}")

if __name__=="__main__": main()
