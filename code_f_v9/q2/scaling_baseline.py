from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from common.metrics import regression_metrics


@dataclass
class ScalingParams:
    E: float
    A: float
    B: float
    alpha: float
    beta: float

    def to_dict(self):
        return asdict(self)


def predict_scaling(n_B, d_B, p: ScalingParams | dict):
    if isinstance(p, dict): p=ScalingParams(**{k:float(p[k]) for k in ["E","A","B","alpha","beta"]})
    n=np.asarray(n_B,float); d=np.asarray(d_B,float)
    return p.E + p.A*np.power(np.maximum(n,1e-12),-p.alpha) + p.B*np.power(np.maximum(d,1e-12),-p.beta)


def _residual(theta, n, d, y):
    E,A,B,alpha,beta=theta
    return predict_scaling(n,d,ScalingParams(E,A,B,alpha,beta))-y


def fit_scaling_law(df: pd.DataFrame, n_starts: int=50, seed: int=42, max_nfev: int=30000, show_progress: bool=True) -> tuple[ScalingParams, dict]:
    x=df[["N_params_B","D_tokens_B","val_loss"]].dropna().copy()
    x=x[(x.N_params_B>0)&(x.D_tokens_B>0)]
    if len(x)<8: raise ValueError("Too few rows to fit scaling law")
    n=x.N_params_B.to_numpy(float); d=x.D_tokens_B.to_numpy(float); y=x.val_loss.to_numpy(float)
    rng=np.random.default_rng(seed)
    upper_E=max(1e-6,0.99*float(np.min(y)))
    lb=np.array([0.0,1e-10,1e-10,1e-4,1e-4])
    ub=np.array([max(upper_E,1e-5),1e4,1e4,2.0,2.0])
    starts=[]
    for _ in range(n_starts):
        E0=rng.uniform(0,max(upper_E*0.95,1e-6))
        A0=np.exp(rng.uniform(np.log(1e-3),np.log(50)))
        B0=np.exp(rng.uniform(np.log(1e-3),np.log(50)))
        a0=rng.uniform(0.05,0.8); b0=rng.uniform(0.05,0.8)
        starts.append([E0,A0,B0,a0,b0])
    # deterministic reasonable start
    starts.insert(0,[min(y)*0.5,1.0,1.0,0.2,0.2])
    best=None
    for x0 in tqdm(starts, desc="Q2 scaling: multi-start NLS", unit="start", leave=False, disable=not show_progress):
        x0=np.minimum(np.maximum(x0,lb+1e-12),ub-1e-12)
        try:
            res=least_squares(_residual,x0,args=(n,d,y),bounds=(lb,ub),loss="soft_l1",f_scale=0.05,max_nfev=max_nfev)
        except Exception:
            continue
        obj=float(np.sum(res.fun**2))
        if best is None or obj<best[0]: best=(obj,res)
    if best is None: raise RuntimeError("All scaling-law fits failed")
    res=best[1]; p=ScalingParams(*map(float,res.x))
    meta={"objective_sse":best[0],"n_rows":int(len(x)),"success":bool(res.success),"message":str(res.message),"nfev":int(res.nfev)}
    return p,meta


def choose_groups(df: pd.DataFrame) -> pd.Series:
    if "record_id" in df.columns:
        rid=df["record_id"].astype(str)
        nun=rid.nunique()
        if 2 <= nun <= max(2,int(0.5*len(df))): return rid
    # Fallback: model size approximates run/model identity.
    return df["N_params_B"].round(6).astype(str)


def grouped_cv(df: pd.DataFrame, n_splits: int=5, seed: int=42, n_starts: int=20):
    x=df.dropna(subset=["N_params_B","D_tokens_B","val_loss"]).copy()
    groups=choose_groups(x)
    ngrp=groups.nunique()
    if ngrp<2: return pd.DataFrame()
    ns=min(n_splits,ngrp)
    gkf=GroupKFold(n_splits=ns)
    rows=[]
    splits = list(gkf.split(x, groups=groups))
    for fold,(tr,te) in enumerate(tqdm(splits, desc="Q2 scaling: GroupKFold", unit="fold", leave=False),start=1):
        p,_=fit_scaling_law(x.iloc[tr],n_starts=n_starts,seed=seed+fold,show_progress=False)
        pred=predict_scaling(x.iloc[te].N_params_B,x.iloc[te].D_tokens_B,p)
        rows.append({"fold":fold,**regression_metrics(x.iloc[te].val_loss,pred),**p.to_dict()})
    return pd.DataFrame(rows)


def evaluate_source(df: pd.DataFrame,p:ScalingParams,source_id:str):
    x=df[df.source_id.eq(source_id)].dropna(subset=["N_params_B","D_tokens_B","val_loss"])
    if len(x)==0: return {"source_id":source_id,"n":0}
    pred=predict_scaling(x.N_params_B,x.D_tokens_B,p)
    return {"source_id":source_id,**regression_metrics(x.val_loss,pred)}
