from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import spearmanr
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline

from common.io_utils import ensure_dir, read_csv, write_csv, write_json, dump_joblib
from common.metrics import regression_metrics
from common.compositional import clr, aitchison_center
from common.plotting import setup_style, savefig, auto_break_barplot
from common.random_utils import set_seed
from q2.scaling_baseline import ScalingParams, fit_scaling_law, predict_scaling, grouped_cv, evaluate_source
from q2.neural_models import (BoundedMonotonicQualityNet, LegacyMonotonicQualityNet,
                              ScaleConditionedMonotonicQualityNet, ScaleConditionedUnconstrainedQualityNet,
                              MLPCompositionNet, AttentionCompositionNet,
                              MixtureLossWrapper, train_quality_net, finetune_quality_loss_net, train_quality_loss_net,
                              train_mixture_net, scaling_with_quality_torch, quality_delta,
                              build_quality_correction_observations)


def source(df, sid):
    return df[df.source_id.eq(sid)].dropna(subset=["N_params_B","D_tokens_B","val_loss"]).copy()


def fit_additive_q(train, params):
    base=predict_scaling(train.N_params_B,train.D_tokens_B,params)
    x=(1-train.Q_score.to_numpy(float)); y=train.val_loss.to_numpy(float)-base
    den=np.sum(x*x); a=max(0.0,float(np.sum(x*y)/(den+1e-12)))
    return a


def pred_additive_q(df,params,a):
    return predict_scaling(df.N_params_B,df.D_tokens_B,params)+a*(1-df.Q_score.to_numpy(float))


def fit_parametric_q(train,params):
    n=train.N_params_B.to_numpy(float); d=train.D_tokens_B.to_numpy(float); q=train.Q_score.to_numpy(float); y=train.val_loss.to_numpy(float)
    def obj(x):
        a=float(x[0]); de=d*np.exp(a*(q-1)); p=ScalingParams(**params.to_dict());
        return predict_scaling(n,de,p)-y
    r=least_squares(obj,[0.5],bounds=([0],[20]),loss="soft_l1",f_scale=.05,max_nfev=10000)
    return float(r.x[0])


def pred_parametric_q(df,params,a):
    de=df.D_tokens_B.to_numpy(float)*np.exp(a*(df.Q_score.to_numpy(float)-1))
    return predict_scaling(df.N_params_B.to_numpy(float),de,params)


def fit_parametric_qd(train,params,init_slope=1.0):
    """Positive Q effect whose slope varies with log D.

    Delta_Q(Q,D) = -softplus(c0 + c1 z_D) * (1-Q),
    z_D = (log D - mean)/std.
    """
    n=train.N_params_B.to_numpy(float); d=train.D_tokens_B.to_numpy(float)
    q=train.Q_score.to_numpy(float); y=train.val_loss.to_numpy(float)
    logd=np.log(np.maximum(d,1e-12)); mu=float(logd.mean()); sd=float(logd.std(ddof=0)); sd=max(sd,1e-6)
    z=(logd-mu)/sd
    def softplus_np(x): return np.logaddexp(0.0,x)
    raw0=np.log(np.expm1(max(float(init_slope),1e-6))) if init_slope < 50 else float(init_slope)
    def resid(v):
        slope=softplus_np(v[0]+v[1]*z); delta=-slope*(1.0-q)
        de=d*np.exp(delta)
        return predict_scaling(n,de,params)-y
    r=least_squares(resid,[raw0,0.0],bounds=([-12,-6],[12,6]),loss="soft_l1",f_scale=.05,max_nfev=20000)
    slope0=float(softplus_np(r.x[0]))
    return {"raw_intercept":float(r.x[0]),"d_coef":float(r.x[1]),"logd_mean":mu,"logd_std":sd,"slope_at_mean_D":slope0}


def pred_parametric_qd(df,params,cfg):
    d=df.D_tokens_B.to_numpy(float); q=df.Q_score.to_numpy(float)
    z=(np.log(np.maximum(d,1e-12))-cfg["logd_mean"])/cfg["logd_std"]
    slope=np.logaddexp(0.0,cfg["raw_intercept"]+cfg["d_coef"]*z)
    de=d*np.exp(-slope*(1.0-q))
    return predict_scaling(df.N_params_B.to_numpy(float),de,params)


def fit_q_spline(train,params):
    base=predict_scaling(train.N_params_B,train.D_tokens_B,params)
    residual=train.val_loss.to_numpy(float)-base
    pipe=make_pipeline(SplineTransformer(n_knots=5,degree=3,include_bias=False),RidgeCV(alphas=np.logspace(-4,3,30)))
    pipe.fit(train[["Q_score"]],residual)
    return pipe


def pred_q_spline(df,params,pipe):
    return predict_scaling(df.N_params_B,df.D_tokens_B,params)+pipe.predict(df[["Q_score"]])


def quality_net_pred(df,params,net,device="cpu"):
    net=net.to(device).eval()
    with torch.no_grad():
        n=torch.tensor(df.N_params_B.to_numpy(float),dtype=torch.float32,device=device)
        d=torch.tensor(df.D_tokens_B.to_numpy(float),dtype=torch.float32,device=device)
        q=torch.tensor(df.Q_score.to_numpy(float),dtype=torch.float32,device=device)
        return scaling_with_quality_torch(n,d,q,params.to_dict(),net).cpu().numpy()


def fit_p_additive(train_p,y):
    X=clr(train_p); m=LinearRegression().fit(X,y); return m


def p_additive_pred(model,p): return model.predict(clr(p))


def fit_p_parametric(train_p,y,beta,p_ref):
    X=clr(train_p)-clr(np.asarray(p_ref).reshape(1,-1))
    # Jointly fit theta and positive a/c using robust least squares.
    d=X.shape[1]
    def resid(v):
        theta=v[:d]; a=np.log1p(np.exp(v[d])); c=np.log1p(np.exp(v[d+1])); delta=X@theta
        return a+c*np.exp(-beta*delta)-y
    r=least_squares(resid,np.zeros(d+2),loss="soft_l1",f_scale=.05,max_nfev=30000)
    return r.x


def pred_p_parametric(v,p,beta,p_ref):
    X=clr(p)-clr(np.asarray(p_ref).reshape(1,-1)); d=X.shape[1]; theta=v[:d]
    a=np.log1p(np.exp(v[d])); c=np.log1p(np.exp(v[d+1])); return a+c*np.exp(-beta*(X@theta))


def eval_p_model(panel,p_cols,loss_cols,name,predict_fn):
    rows=[]
    for (pair,split,scale),g in panel.groupby(["pair_id","split","model_scale_label"]):
        y=g[loss_cols].mean(axis=1).to_numpy(float); pred=np.asarray(predict_fn(g[p_cols].to_numpy(float)),float)
        m=regression_metrics(y,pred)
        rows.append({"component":"p_branch","model":name,"pair_id":pair,"split":split,"model_scale_label":scale,**m})
    return rows


def save_q_checkpoint(net,path,architecture="scale_conditioned_monotonic_qd_v3",**extra):
    payload={"state_dict":net.cpu().state_dict(),"architecture":architecture}
    if isinstance(net,ScaleConditionedMonotonicQualityNet):
        payload.update({
            "hidden":net.hidden,"d_hidden":net.d_hidden,"n_basis":net.n_basis,
            "s_max":net.s_max,"soft_threshold":net.soft_threshold,
            "init_slope":net.init_slope,"init_d_coef":net.init_d_coef,
            "logd_mean":float(net.logd_mean.cpu()),"logd_std":float(net.logd_std.cpu()),
        })
    elif isinstance(net,BoundedMonotonicQualityNet):
        payload.update({"hidden":net.hidden,"s_max":net.s_max,"soft_threshold":net.soft_threshold,
                        "init_slope":net.init_slope,"anchored":net.anchored,"monotonic":net.monotonic})
    elif isinstance(net,ScaleConditionedUnconstrainedQualityNet):
        payload.update({"hidden":net.hidden,"s_max":net.s_max,
                        "logd_mean":float(net.logd_mean.cpu()),"logd_std":float(net.logd_std.cpu())})
    payload.update(extra); torch.save(payload,path)

def save_p_checkpoint(net,path,p_ref,**cfg):
    torch.save({"state_dict":net.cpu().state_dict(),"p_ref":np.asarray(p_ref,float).tolist(),**cfg},path)


def main():
    ap=argparse.ArgumentParser(description="Q2 scaling-law baselines, CE-ADNSL main model, and ablations.")
    ap.add_argument("--processed",type=Path,required=True)
    ap.add_argument("--q1",type=Path,required=True,help="Directory containing q1_mixture outputs")
    ap.add_argument("--out",type=Path,default=Path("results/q2"))
    ap.add_argument("--epochs-q",type=int,default=1500)
    ap.add_argument("--epochs-p",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fast",action="store_true",help="Smoke-test settings: fewer starts/epochs.")
    args=ap.parse_args(); set_seed(args.seed); setup_style()
    if args.fast: args.epochs_q=min(args.epochs_q,80); args.epochs_p=min(args.epochs_p,100)
    out=ensure_dir(args.out); figs=ensure_dir(out/"figures"); art=ensure_dir(out/"artifacts")

    sc=read_csv(args.processed/"B"/"q2_scaling.csv",required=["source_id","N_params_B","D_tokens_B","val_loss"])
    for c in ["N_params_B","D_tokens_B","val_loss","Q_score"]:
        if c in sc: sc[c]=pd.to_numeric(sc[c],errors="coerce")
    b1=source(sc,"B1"); starts=8 if args.fast else 50
    params,meta=fit_scaling_law(b1,n_starts=starts,seed=args.seed)
    write_json(params.to_dict(),art/"scaling_params.json"); write_json(meta,out/"q2_scaling_fit_meta.json")
    cv=grouped_cv(b1,n_splits=5,seed=args.seed,n_starts=5 if args.fast else 20); write_csv(cv,out/"q2_scaling_groupcv.csv")
    ext=pd.DataFrame([evaluate_source(sc,params,s) for s in ["B1","B2","B3","B4","B5","B10"]])
    write_csv(ext,out/"q2_scaling_external_validation.csv")

    # B11 auxiliary metadata is used for cross-family residual diagnostics, not as a causal predictor.
    fam=sc.dropna(subset=["N_params_B","D_tokens_B","val_loss"]).copy()
    fam["classic_pred"]=predict_scaling(fam.N_params_B,fam.D_tokens_B,params); fam["classic_residual"]=fam.val_loss-fam.classic_pred
    meta_cols=[c for c in ["family_metadata_repo_count","family_metadata_weight_file_count","family_metadata_weight_total_bytes","family_metadata_config_bytes_mean","family_metadata_readme_bytes_mean"] if c in fam.columns]
    group_cols=[c for c in ["family_key","model_family"] if c in fam.columns]
    if group_cols:
        agg_spec={"n_rows":("classic_residual","size"),"mean_residual":("classic_residual","mean"),"rmse_residual":("classic_residual",lambda x:float(np.sqrt(np.mean(np.asarray(x,float)**2))))}
        for c in meta_cols: agg_spec[c]=(c,"first")
        fdiag=fam.groupby(group_cols,dropna=False).agg(**agg_spec).reset_index(); write_csv(fdiag,out/"q2_family_metadata_residual_diagnostics.csv")

    # ----- Q branch comparisons -----
    b6=source(sc,"B6").dropna(subset=["Q_score"]); b7=source(sc,"B7").dropna(subset=["Q_score"]); b8=source(sc,"B8").dropna(subset=["Q_score"])
    if len(b6)==0: raise ValueError("B6 quality data not found")

    # Traditional baselines remain on the original Loss scale.
    qa=fit_additive_q(b6,params); qpar=fit_parametric_q(b6,params); qspline=fit_q_spline(b6,params)
    dump_joblib(qspline,art/"q_spline.joblib")

    # Residual-supervision labels implied by the fitted classic Scaling Law.
    b6q=build_quality_correction_observations(b6,params.to_dict())
    b7q=build_quality_correction_observations(b7,params.to_dict()) if len(b7) else b7.copy()
    b8q=build_quality_correction_observations(b8,params.to_dict()) if len(b8) else b8.copy()
    qobs=pd.concat([x.assign(source_id=sid) for sid,x in [("B6",b6q),("B7",b7q),("B8",b8q)] if len(x)],ignore_index=True,sort=False)
    keep=[c for c in ["source_id","record_id","N_params_B","D_tokens_B","Q_score","val_loss","scaling_nondata_term","observed_data_term","classic_data_term","delta_Q_obs","q_correction_valid","q_correction_physical"] if c in qobs.columns]
    write_csv(qobs[keep],out/"q2_q_correction_observations.csv")
    vrows=[]
    for sid,g in [("B6",b6q),("B7",b7q),("B8",b8q)]:
        if not len(g): continue
        valid=g["q_correction_valid"].astype(bool)
        phys=g["q_correction_physical"].astype(bool)
        vrows.append({
            "source_id":sid,
            "n_total":int(len(g)),
            "n_valid":int(valid.sum()),
            "n_invalid":int((~valid).sum()),
            "valid_rate":float(valid.mean()),
            "n_physical_delta_le_0":int(phys.sum()),
            "physical_rate_among_valid":float(phys.sum()/max(1,valid.sum())),
            "delta_obs_min":float(g.loc[valid,"delta_Q_obs"].min()) if valid.any() else np.nan,
            "delta_obs_median":float(g.loc[valid,"delta_Q_obs"].median()) if valid.any() else np.nan,
            "delta_obs_max":float(g.loc[valid,"delta_Q_obs"].max()) if valid.any() else np.nan,
        })
    write_csv(pd.DataFrame(vrows),out/"q2_q_correction_validity.csv")

    # Main Q branch v3: scale-conditioned monotonic quality network.
    # B6/B7 provide joint (N,D,Q,Loss) observations, so Delta_Q is allowed to
    # depend on log D while remaining monotone in Q for every fixed D.
    qparqd=fit_parametric_qd(b6,params,init_slope=qpar)
    q_s_max=8.0; q_soft_threshold=4.0
    qnet=ScaleConditionedMonotonicQualityNet(
        hidden=8,d_hidden=8,n_basis=6,s_max=q_s_max,soft_threshold=q_soft_threshold,
        init_slope=qparqd["slope_at_mean_D"],init_d_coef=qparqd["d_coef"],
        logd_mean=qparqd["logd_mean"],logd_std=qparqd["logd_std"]
    )
    qnet,qhist_a,qep_a=train_quality_net(
        qnet,b6q,b7q,args.epochs_q,lr=5e-4,patience=180,seed=args.seed,device=args.device,
        physical_only=True,select_params=params.to_dict(),select_valid_raw=b7
    )
    qnet,qhist_b,qep_b=finetune_quality_loss_net(
        qnet,params.to_dict(),b6,b7,epochs=max(300,args.epochs_q//2),lr=1e-4,
        patience=140,seed=args.seed+1,device=args.device
    )
    save_q_checkpoint(qnet,art/"q_branch.pt",architecture="scale_conditioned_monotonic_qd_v3")

    # Ablation: remove D conditioning but keep bounded monotonicity.
    qnod=BoundedMonotonicQualityNet(hidden=8,s_max=6.0,soft_threshold=3.0,
                                    anchored=True,monotonic=True,init_slope=qpar)
    qnod,_,_=train_quality_net(qnod,b6q,b7q,args.epochs_q,lr=5e-4,patience=150,seed=args.seed+10,
                               device=args.device,physical_only=True,select_params=params.to_dict(),select_valid_raw=b7)
    qnod,_,_=finetune_quality_loss_net(qnod,params.to_dict(),b6,b7,epochs=max(300,args.epochs_q//2),lr=1e-4,
                                       patience=120,seed=args.seed+11,device=args.device)

    # Ablation: use Q and D but remove monotonicity.
    qun=ScaleConditionedUnconstrainedQualityNet(hidden=16,s_max=q_s_max,
                                                logd_mean=qparqd["logd_mean"],logd_std=qparqd["logd_std"])
    qun,_,_=train_quality_net(qun,b6q,b7q,args.epochs_q,lr=3e-4,patience=150,seed=args.seed+20,
                              device=args.device,physical_only=False,select_params=params.to_dict(),select_valid_raw=b7)
    qun,_,_=finetune_quality_loss_net(qun,params.to_dict(),b6,b7,epochs=max(300,args.epochs_q//2),lr=1e-4,
                                      patience=120,seed=args.seed+21,device=args.device)

    # Diagnostic only: retain the original unbounded Q-only architecture.
    qlegacy=LegacyMonotonicQualityNet(hidden=16,anchored=True,monotonic=True)
    qlegacy,_,_=train_quality_loss_net(qlegacy,params.to_dict(),b6,b7,args.epochs_q,seed=args.seed+30,device=args.device)

    qrows=[]
    for sid,g in [("B6",b6),("B7",b7),("B8",b8)]:
        if len(g)==0: continue
        candidates={
            "SL0_classic":predict_scaling(g.N_params_B,g.D_tokens_B,params),
            "SL1_additive_Q":pred_additive_q(g,params,qa),
            "SL2_parametric_Deff_Q":pred_parametric_q(g,params,qpar),
            "SL2b_parametric_Deff_QxD":pred_parametric_qd(g,params,qparqd),
            "Q_spline_baseline":pred_q_spline(g,params,qspline),
            "SL3_4_scale_conditioned_monotonic_QD":quality_net_pred(g,params,qnet,args.device),
            "ablation_no_D_conditioning_Q":quality_net_pred(g,params,qnod,args.device),
            "ablation_unconstrained_QD":quality_net_pred(g,params,qun,args.device),
            "ablation_legacy_unbounded_monotonic_Q":quality_net_pred(g,params,qlegacy,args.device),
        }
        for name,pred in candidates.items():
            qrows.append({"component":"q_branch","source_id":sid,"model":name,**regression_metrics(g.val_loss,pred)})
    qmetrics=pd.DataFrame(qrows); write_csv(qmetrics,out/"q2_q_branch_comparison.csv")
    median_d=float(np.median(b6.D_tokens_B))
    with torch.no_grad(): median_slope=float(qnet.slope(torch.tensor([median_d],dtype=torch.float32,device=args.device))[0].cpu())
    write_json({
        "q_additive_a":qa,
        "q_parametric_a":qpar,
        "q_parametric_qd":qparqd,
        "q_architecture":"scale_conditioned_monotonic_quality_network",
        "q_formula":"Delta_Q(Q,D)=-softcap[s(D)(1-Q)+D-conditioned monotone basis residual]",
        "q_hidden":8,"q_d_hidden":8,"q_n_basis":6,
        "q_s_max":q_s_max,"q_soft_threshold":q_soft_threshold,
        "q_final_slope_at_median_D":median_slope,
        "q_residual_pretrain_best_epoch":qep_a,
        "q_loss_finetune_best_epoch":qep_b,
        "q_residual_learning_rate":5e-4,
        "q_finetune_learning_rate":1e-4,
        "q_gradient_clip":1.0,
        "q_checkpoint_rule":"B7 final-loss RMSE with epoch-0 QxD-parametric initialization fallback",
    },out/"q2_q_branch_parameters.json")
    qhist=qhist_a+qhist_b
    write_csv(pd.DataFrame(qhist,columns=["stage","epoch","train_objective","valid_objective","selection_rmse","learning_rate"]),out/"q2_q_branch_training_history.csv")

    # Monotonicity/boundedness diagnostics at low/median/high D.
    d_levels=np.quantile(b6.D_tokens_B.to_numpy(float),[0.1,0.5,0.9])
    mono=[]
    for dval in d_levels:
        for name,net in [("scale_conditioned_monotonic_QD",qnet),("no_D_conditioning_Q",qnod),
                         ("unconstrained_QD",qun),("legacy_unbounded_Q",qlegacy)]:
            net=net.to(args.device).eval()
            qgrid=torch.linspace(0,1,201,requires_grad=True,device=args.device)
            dgrid=torch.full_like(qgrid,float(dval))
            vals=quality_delta(net,qgrid,dgrid)
            grad=torch.autograd.grad(vals.sum(),qgrid,retain_graph=False)[0].detach().cpu().numpy()
            vv=vals.detach().cpu().numpy()
            mono.append({"model":name,"D_tokens_B":float(dval),"min_delta_Q":float(np.min(vv)),
                         "max_delta_Q":float(np.max(vv)),"delta_at_Q1":float(vv[-1]),
                         "min_dDelta_dQ":float(grad.min()),"violation_rate":float(np.mean(grad < -1e-8))})
    write_csv(pd.DataFrame(mono),out/"q2_q_monotonicity.csv")

    # Explicit scale-conditioning surface for interpretation and later plots.
    q_surface=np.linspace(max(0.01,float(b6.Q_score.min())),1.0,80)
    d_surface=np.exp(np.linspace(np.log(max(1e-12,float(b6.D_tokens_B.min()))),
                                np.log(float(b8.D_tokens_B.max() if len(b8) else b6.D_tokens_B.max())),60))
    surface_rows=[]
    qnet=qnet.to(args.device).eval()
    for dval in d_surface:
        tq=torch.tensor(q_surface,dtype=torch.float32,device=args.device,requires_grad=True)
        td=torch.full_like(tq,float(dval)); vals=qnet(tq,td)
        gdq=torch.autograd.grad(vals.sum(),tq)[0].detach().cpu().numpy(); vv=vals.detach().cpu().numpy()
        for qv,dv,gg in zip(q_surface,vv,gdq):
            surface_rows.append({"Q":float(qv),"D_tokens_B":float(dval),"delta_Q":float(dv),
                                 "dDelta_dQ":float(gg),"effective_data_multiplier":float(np.exp(dv))})
    qsurface=pd.DataFrame(surface_rows); write_csv(qsurface,out/"q2_q_scale_conditioning_surface.csv")

    # Direct diagnostic: observed correction, coloured by D, plus fitted curves at D quantiles.
    fig,ax=plt.subplots(figsize=(9,5.5))
    qobs_valid=qobs[qobs["q_correction_valid"].astype(bool)].copy()
    if len(qobs_valid):
        qobs_valid["log10_D_tokens_B"]=np.log10(np.maximum(qobs_valid["D_tokens_B"].to_numpy(float),1e-12))
        sns.scatterplot(data=qobs_valid,x="Q_score",y="delta_Q_obs",hue="log10_D_tokens_B",palette="viridis",alpha=.5,s=24,ax=ax)
        qline=np.linspace(max(0.0,float(qobs_valid.Q_score.min())),min(1.0,float(qobs_valid.Q_score.max())),200)
        for dval,lab in zip(d_levels,["D p10","D p50","D p90"]):
            with torch.no_grad():
                tq=torch.tensor(qline,dtype=torch.float32,device=args.device); td=torch.full_like(tq,float(dval))
                predline=qnet(tq,td).cpu().numpy()
            ax.plot(qline,predline,linewidth=2,label=f"{lab}={dval:.3g}B")
        ax.axhline(0,color="gray",linewidth=1,linestyle="--"); ax.legend()
    ax.set_title("Scale-conditioned quality correction: observed and fitted")
    ax.set_ylabel(r"Observed / fitted $\Delta_Q(Q,D)$")
    savefig(figs/"q2_02a_q_correction_observed.png",fig)

    piv=qsurface.pivot_table(index="D_tokens_B",columns="Q",values="delta_Q",aggfunc="mean")
    fig,ax=plt.subplots(figsize=(10,6)); sns.heatmap(piv,cmap="viridis",ax=ax,xticklabels=False,yticklabels=False)
    xi=np.unique(np.linspace(0,len(piv.columns)-1,8).round().astype(int)); yi=np.unique(np.linspace(0,len(piv.index)-1,8).round().astype(int))
    ax.set_xticks(xi+0.5); ax.set_xticklabels([f"{float(piv.columns[i]):.2f}" for i in xi],rotation=0)
    def _fmt_d(v):
        v=float(v)
        return f"{v:.2f}" if v<1 else (f"{v:.1f}" if v<100 else f"{v:.0f}")
    ax.set_yticks(yi+0.5); ax.set_yticklabels([_fmt_d(piv.index[i]) for i in yi],rotation=0)
    ax.set_title(r"Scale-conditioned $\Delta_Q(Q,D)$ surface"); ax.set_xlabel("Quality Q"); ax.set_ylabel("Training tokens D (B)")
    savefig(figs/"q2_02b_q_scale_conditioning_heatmap.png",fig)

    # ----- p branch comparisons -----
    mix=read_csv(args.processed/"A"/"q1_mixture_loss.csv",required=["pair_id","split","model_scale_label"])
    p_cols=[c for c in mix.columns if c.startswith("p_")]; loss_cols=[c for c in mix.columns if c.startswith("loss_") and c != "loss_nature"]
    for c in p_cols+loss_cols: mix[c]=pd.to_numeric(mix[c],errors="coerce")
    mix=mix.dropna(subset=p_cols+loss_cols)
    tr=mix[mix.pair_id.eq("A4_A5")].copy(); y=tr[loss_cols].mean(axis=1).to_numpy(float); P=tr[p_cols].to_numpy(float); pref=aitchison_center(P)
    rng=np.random.default_rng(args.seed); idx=np.arange(len(tr)); rng.shuffle(idx); cut=max(1,int(.8*len(idx))); it,iv=idx[:cut],idx[cut:]
    padd=fit_p_additive(P[it],y[it]); pparam=fit_p_parametric(P[it],y[it],params.beta,pref)
    dump_joblib(padd,art/"p_additive_clr.joblib"); np.save(art/"p_parametric.npy",pparam)

    pmlp=MLPCompositionNet(n_parts=17,use_clr=True,anchored=True,p_ref=pref); wmlp=MixtureLossWrapper(pmlp,params.beta)
    wmlp,_,_=train_mixture_net(wmlp,P[it],y[it],P[iv],y[iv],args.epochs_p,seed=args.seed,device=args.device)
    patt=AttentionCompositionNet(n_parts=17,d_model=32,n_heads=4,n_layers=2,d_ff=64,dropout=.1,use_clr=True,use_embedding=True,anchored=True,p_ref=pref); watt=MixtureLossWrapper(patt,params.beta)
    watt,phist,pep=train_mixture_net(watt,P[it],y[it],P[iv],y[iv],args.epochs_p,seed=args.seed+1,device=args.device)
    # ablations: raw-p attention and no domain embedding
    praw=AttentionCompositionNet(17,32,4,2,64,.1,False,True,True,pref); wraw=MixtureLossWrapper(praw,params.beta); wraw,_,_=train_mixture_net(wraw,P[it],y[it],P[iv],y[iv],args.epochs_p,seed=args.seed+2,device=args.device)
    pnoemb=AttentionCompositionNet(17,32,4,2,64,.1,True,False,True,pref); wnoemb=MixtureLossWrapper(pnoemb,params.beta); wnoemb,_,_=train_mixture_net(wnoemb,P[it],y[it],P[iv],y[iv],args.epochs_p,seed=args.seed+3,device=args.device)

    save_p_checkpoint(patt,art/"p_branch.pt",pref,n_parts=17,d_model=32,n_heads=4,n_layers=2,d_ff=64,dropout=.1,use_clr=True,use_embedding=True,anchored=True)
    write_csv(pd.DataFrame(phist,columns=["epoch","train_huber","valid_huber"]),out/"q2_p_branch_training_history.csv")
    write_json({"p_reference":pref.tolist(),"p_columns":p_cols,"p_best_epoch":pep},art/"model_metadata.json")

    def torchpred(wrapper,Px):
        wrapper=wrapper.to(args.device).eval()
        with torch.no_grad(): return wrapper(torch.tensor(Px,dtype=torch.float32,device=args.device)).cpu().numpy()
    prows=[]
    predictors={
        "SL1_additive_CLR":lambda x:p_additive_pred(padd,x),
        "SL2_parametric_Deff_CLR":lambda x:pred_p_parametric(pparam,x,params.beta,pref),
        "SL3_neural_Deff_CLR_MLP":lambda x:torchpred(wmlp,x),
        "SL4_CE_ADNSL_attention":lambda x:torchpred(watt,x),
        "ablation_raw_p_attention":lambda x:torchpred(wraw,x),
        "ablation_no_embedding":lambda x:torchpred(wnoemb,x),
    }
    for name,fn in predictors.items(): prows += eval_p_model(mix,p_cols,loss_cols,name,fn)
    pmetrics=pd.DataFrame(prows)
    # Bring Q1 mixture baselines into the same p-branch scorecard when available.
    q1cmp=args.q1/"q1_mixture_model_comparison.csv"
    if q1cmp.exists():
        q1m=pd.read_csv(q1cmp); q1m=q1m[q1m["target"].eq("loss_macro")].copy(); q1m["component"]="p_branch"; q1m["model"]="Q1_"+q1m["method"].astype(str); q1m["source_id"]=q1m["pair_id"]
        pmetrics=pd.concat([pmetrics,q1m],ignore_index=True,sort=False)
    write_csv(pmetrics,out/"q2_p_branch_comparison.csv")

    # Mean attention matrix over training recipes. Last encoder layer, average over heads and samples.
    patt=patt.to(args.device).eval(); tP=torch.tensor(P,dtype=torch.float32,device=args.device)
    with torch.no_grad(): att,pool=patt.attention_maps(tP); A=att[-1].mean(dim=(0,1)).cpu().numpy(); poolm=pool.mean(0).cpu().numpy()
    adom=[c.removeprefix("p_") for c in p_cols]
    write_csv(pd.DataFrame(A,index=adom,columns=adom).reset_index().rename(columns={"index":"domain"}),out/"q2_attention_matrix.csv")
    write_csv(pd.DataFrame({"domain":adom,"attention_pooling_weight":poolm}),out/"q2_attention_pooling.csv")

    # Domain-transfer marginal effects on the simplex around p_ref.
    delta=1e-4; transfer=np.full((17,17),np.nan); qrep=float(np.median(b6.Q_score)); Nrep=float(np.median(b1.N_params_B))*1e9; Drep=float(np.median(b1.D_tokens_B))*1e9
    qnet=qnet.to(args.device).eval(); patt=patt.to(args.device).eval()
    with torch.no_grad():
        tqrep=torch.tensor([qrep],dtype=torch.float32,device=args.device)
        tdrep=torch.tensor([Drep/1e9],dtype=torch.float32,device=args.device)
        dqrep=float(qnet(tqrep,tdrep)[0].cpu())
    def final_loss_for_p(pp):
        with torch.no_grad(): dp=float(patt(torch.tensor(pp.reshape(1,-1),dtype=torch.float32,device=args.device))[0].cpu())
        return params.E+params.A*(Nrep/1e9)**(-params.alpha)+params.B*((Drep/1e9)*np.exp(dqrep+dp))**(-params.beta)
    for i in range(17):
        for j in range(17):
            if i==j or pref[j]<=delta: continue
            pp=pref.copy(); pp[i]+=delta; pp[j]-=delta; transfer[i,j]=(final_loss_for_p(pp)-final_loss_for_p(pref))/delta
    write_csv(pd.DataFrame(transfer,index=adom,columns=adom).reset_index().rename(columns={"index":"to_domain"}),out/"q2_domain_transfer_marginal_matrix.csv")

    # Scheffe-vs-attention concordance is interpretive, not causal.
    interpath=args.q1/"q1_domain_interaction_matrix.csv"
    if interpath.exists():
        si=pd.read_csv(interpath,index_col=0); si.index=[str(x).removeprefix("p_") for x in si.index]; si.columns=[str(x).removeprefix("p_") for x in si.columns]
        common=[d for d in adom if d in si.index and d in si.columns]; vals=[]
        for ii in range(len(common)):
            for jj in range(ii+1,len(common)):
                di,dj=common[ii],common[jj]; ai=(A[adom.index(di),adom.index(dj)]+A[adom.index(dj),adom.index(di)])/2; vals.append({"domain_i":di,"domain_j":dj,"scheffe_gamma":float(si.loc[di,dj]),"abs_gamma":abs(float(si.loc[di,dj])),"attention":float(ai)})
        conc=pd.DataFrame(vals); write_csv(conc,out/"q2_scheffe_attention_concordance.csv")
        if len(conc): write_json({"spearman_abs_gamma_attention":float(conc[["abs_gamma","attention"]].corr(method="spearman").iloc[0,1])},out/"q2_scheffe_attention_summary.json")

    # Unified E3/E4 scorecard is deliberately component-wise; no fake joint test set is created.
    score=pd.concat([qmetrics,pmetrics],ignore_index=True,sort=False); write_csv(score,out/"q2_model_comparison_scorecard.csv")

    # Elasticities and Q<->N equivalence on a representative grid using p_ref.
    # Because Delta_Q now depends on D, dL/dD must include the learned Q-D interaction.
    # Autograd is used so this total derivative is computed correctly.
    N0=float(np.median(b1.N_params_B))*1e9; D0=float(np.median(b1.D_tokens_B))*1e9
    qvals=np.linspace(max(0.05,float(b6.Q_score.min())),min(1.0,float(b8.Q_score.max() if len(b8) else b6.Q_score.max())),80)
    qnet=qnet.to(args.device).eval(); nB=N0/1e9; dB=D0/1e9
    nt=torch.full((len(qvals),),float(nB),dtype=torch.float32,device=args.device,requires_grad=True)
    dt=torch.full((len(qvals),),float(dB),dtype=torch.float32,device=args.device,requires_grad=True)
    qt=torch.tensor(qvals,dtype=torch.float32,device=args.device,requires_grad=True)
    Lt=scaling_with_quality_torch(nt,dt,qt,params.to_dict(),qnet)
    grad_n,grad_d,grad_q=torch.autograd.grad(Lt.sum(),(nt,dt,qt))
    with torch.no_grad(): dq=qnet(qt.detach(),dt.detach()).cpu().numpy()
    L=Lt.detach().cpu().numpy(); dL_dN=grad_n.detach().cpu().numpy()/1e9; dL_dD=grad_d.detach().cpu().numpy()/1e9; dL_dQ=grad_q.detach().cpu().numpy()
    elast=pd.DataFrame({"Q":qvals,"D_tokens_B":dB,"delta_Q":dq,"L":L,"dL_dN":dL_dN,"dL_dD":dL_dD,"dL_dQ":dL_dQ,
                        "epsilon_N":dL_dN*N0/L,"epsilon_D":dL_dD*D0/L,"epsilon_Q":dL_dQ*qvals/L,
                        "deltaN_equiv_for_dQ_0.1":np.abs(dL_dQ)/(np.abs(dL_dN)+1e-30)*0.1})
    write_csv(elast,out/"q2_elasticity_quality_parameter_tradeoff.csv")

    # Large-model extrapolation range table.
    large_path=args.processed/"B"/"q2_large_models.csv"
    if large_path.exists():
        lg=pd.read_csv(large_path); write_csv(lg,out/"q2_large_model_reference.csv")
        valid=lg.dropna(subset=["N_params_B","D_tokens_B"]).copy()
        if len(valid):
            valid["pred_classic_loss_Q1_pref"]=predict_scaling(valid.N_params_B,valid.D_tokens_B,params)
            # CE-ADNSL reference condition Q=1, p=p_ref is exactly the classic law by construction.
            valid["pred_CE_ADNSL_reference_loss"]=valid["pred_classic_loss_Q1_pref"]
            write_csv(valid,out/"q2_large_model_extrapolation_predictions.csv")

    # Figures
    fig,axes,_=auto_break_barplot(data=ext[ext.source_id.isin(["B1","B2","B4","B5"])],x="source_id",y="rmse",color="#4C72B0",figsize=(8,5),title="Classic scaling-law validation RMSE"); savefig(figs/"q2_01_scaling_external_rmse.png",fig)
    fpath=out/"q2_family_metadata_residual_diagnostics.csv"
    if fpath.exists():
        fd=pd.read_csv(fpath); xcol="family_metadata_weight_total_bytes" if "family_metadata_weight_total_bytes" in fd.columns else None
        if xcol and fd[xcol].notna().sum()>=2:
            fig,ax=plt.subplots(figsize=(8,5)); sns.scatterplot(data=fd,x=xcol,y="rmse_residual",hue="model_family" if "model_family" in fd else None,s=70,ax=ax); ax.set_xscale("log"); ax.set_title("Cross-family Scaling residual vs B11 family metadata"); savefig(figs/"q2_01b_family_metadata_residual.png",fig)
    qplot=qmetrics[qmetrics.source_id.eq("B7")].sort_values("rmse")
    fig,axes,_=auto_break_barplot(data=qplot,x="rmse",y="model",color="#4C72B0",figsize=(9,5),title="Quality branch model comparison on B7",xlabel="RMSE"); savefig(figs/"q2_02_q_branch_comparison.png",fig)
    pp=pmetrics[(pmetrics.pair_id=="A6_A7")].sort_values("rmse")
    fig,axes,_=auto_break_barplot(data=pp,x="rmse",y="model",color="#4C72B0",figsize=(9,6),title="Mixture branch comparison on A6/A7",xlabel="RMSE"); savefig(figs/"q2_03_p_branch_comparison.png",fig)
    fig,ax=plt.subplots(figsize=(11,9)); sns.heatmap(A,cmap="mako",xticklabels=adom,yticklabels=adom,ax=ax); ax.set_title("CE-ADNSL mean self-attention matrix"); savefig(figs/"q2_04_attention_heatmap.png",fig)
    fig,ax=plt.subplots(figsize=(8,5)); sns.lineplot(data=elast,x="Q",y=-elast["epsilon_Q"],label="Quality",ax=ax); sns.lineplot(data=elast,x="Q",y=-elast["epsilon_N"],label="Parameters",ax=ax); sns.lineplot(data=elast,x="Q",y=-elast["epsilon_D"],label="Data",ax=ax); ax.set_ylabel("Benefit elasticity (-epsilon)"); ax.set_title(f"Marginal benefit elasticities at D={D0/1e9:.3g}B tokens"); savefig(figs/"q2_05_elasticities.png",fig)
    fig,ax=plt.subplots(figsize=(8,5)); sns.lineplot(data=elast,x="Q",y="deltaN_equiv_for_dQ_0.1",ax=ax); ax.set_yscale("log"); ax.set_title("Parameter-equivalent gain for ΔQ=0.1"); ax.set_ylabel("Equivalent additional parameters"); savefig(figs/"q2_06_q_to_N_equivalence.png",fig)

    # Basic data-range metadata for Q3 bounds.
    n_all=sc.N_params_B.dropna(); d_all=sc.D_tokens_B.dropna()
    meta2=json.load(open(art/"model_metadata.json","r",encoding="utf-8")); meta2.update({"N_B_min":float(n_all[n_all>0].min()),"N_B_max":float(n_all.max()),"D_B_min":float(d_all[d_all>0].min()),"D_B_max":float(d_all.max()),"quality_train_min":float(b6.Q_score.min()),"quality_test_max":float(b8.Q_score.max() if len(b8) else b6.Q_score.max())})
    write_json(meta2,art/"model_metadata.json")
    print(f"Q2 complete: {out}; device={args.device}")

if __name__=="__main__": main()
