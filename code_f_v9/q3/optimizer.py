from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import torch
from tqdm.auto import tqdm

from common.compositional import softmax_logits_to_composition, composition_entropy
from common.io_utils import read_json
from q2.model_api import load_predictor

ETA=2e-4


def quality_cost_g(Q,kind="exponential"):
    Q=np.asarray(Q,float)
    if kind=="exponential": return 1e7*np.exp(6.0*Q)
    if kind=="power": return 5e9*np.power(Q,4.0)
    if kind=="log": return 2e9*np.log1p(10.0*Q)
    raise ValueError(kind)


def quality_cost_g_prime(Q,kind="exponential"):
    if kind=="exponential": return 6e7*np.exp(6.0*Q)
    if kind=="power": return 20e9*(Q**3)
    if kind=="log": return 20e9/(1+10*Q)
    raise ValueError(kind)


def total_cost(N,D,Q,Q0,Lctx,kind="exponential"):
    train=6.0*N*D
    q=D*max(float(quality_cost_g(Q,kind)-quality_cost_g(Q0,kind)),0.0)
    attn=ETA*N*D*Lctx
    return train+q+attn,train,q,attn


class PredictorEnsemble:
    """Q3-efficient ensemble: one neural correction pair + many bootstrap scaling parameter draws."""
    def __init__(self,q2_dir:Path,bootstrap_csv:Path|None=None,device="cpu",max_samples=300):
        self.base=load_predictor(q2_dir/"artifacts",device=device)
        self.device=device
        self.params=[self.base.params]
        self.full_models=[]
        ensemble_dir=Path(q2_dir)/"bootstrap"/"ensemble"
        if ensemble_dir.exists():
            dirs=sorted([d for d in ensemble_dir.glob("boot_*") if (d/"q_branch.pt").exists() and (d/"p_branch.pt").exists()])
            if len(dirs)>max_samples:
                pick=np.linspace(0,len(dirs)-1,max_samples).round().astype(int); dirs=[dirs[i] for i in pick]
            for d in dirs:
                try:self.full_models.append(load_predictor(d,device=device))
                except Exception:pass
        if not self.full_models and bootstrap_csv and Path(bootstrap_csv).exists():
            b=pd.read_csv(bootstrap_csv).dropna(subset=["E","A","B","alpha","beta"])
            if len(b)>max_samples:
                b=b.sample(max_samples,random_state=42)
            self.params += [r for r in b[["E","A","B","alpha","beta"]].to_dict("records")]
    def predict_matrix(self,N,D,Q,p):
        """Return scenario losses with shape (n_scenarios, n_candidates).

        Q3 optimizers evaluate hundreds of candidates per DE generation.  Batching
        candidates here avoids one PyTorch Attention forward pass per scalar
        candidate, which was the main reason the old core-budget progress bar could
        remain at 0 for minutes.
        """
        N=np.atleast_1d(np.asarray(N,float)); D=np.atleast_1d(np.asarray(D,float)); Q=np.atleast_1d(np.asarray(Q,float))
        P=np.asarray(p,float)
        if P.ndim==1: P=P.reshape(1,-1)
        m=max(len(N),len(D),len(Q),P.shape[0])
        if len(N)==1: N=np.full(m,float(N[0]))
        if len(D)==1: D=np.full(m,float(D[0]))
        if len(Q)==1: Q=np.full(m,float(Q[0]))
        if P.shape[0]==1 and m>1: P=np.repeat(P,m,axis=0)
        if self.full_models:
            preds=[np.asarray(self.base.predict(N,D,Q,P),float).reshape(-1)]
            # Each bootstrap network sees the whole candidate batch at once.
            for model in self.full_models:
                preds.append(np.asarray(model.predict(N,D,Q,P),float).reshape(-1))
            return np.vstack(preds)
        dq,dp=self.base.corrections(Q,P,D=D); s=np.asarray(dq,float)+np.asarray(dp,float)
        n=N/1e9; d=D/1e9
        E=np.asarray([par["E"] for par in self.params],float)[:,None]
        A=np.asarray([par["A"] for par in self.params],float)[:,None]
        B=np.asarray([par["B"] for par in self.params],float)[:,None]
        alpha=np.asarray([par["alpha"] for par in self.params],float)[:,None]
        beta=np.asarray([par["beta"] for par in self.params],float)[:,None]
        return E + A*np.power(np.maximum(n[None,:],1e-12),-alpha) + B*np.power(np.maximum(d[None,:],1e-12)*np.exp(s[None,:]),-beta)
    def predict_samples(self,N,D,Q,p):
        return self.predict_matrix(N,D,Q,p)[:,0]
    def mean(self,N,D,Q,p): return float(np.mean(self.predict_samples(N,D,Q,p)))


@dataclass
class BoundsSpec:
    logN: tuple[float,float]
    logD: tuple[float,float]
    Q: tuple[float,float]
    logit: tuple[float,float]=(-6.0,6.0)


def decode_x(x,joint_p=True,fixed_p=None):
    N=10.0**float(x[0]); D=10.0**float(x[1]); Q=float(x[2])
    p=softmax_logits_to_composition(x[3:]) if joint_p else np.asarray(fixed_p,float)
    return N,D,Q,p


def risk_object(losses,mode="mean_cvar",risk_lambda=.2):
    l=np.asarray(losses,float); mean=float(np.mean(l))
    if mode=="mean": return mean
    if mode=="mean_sd": return mean+risk_lambda*float(np.std(l,ddof=1) if len(l)>1 else 0)
    if mode=="mean_cvar":
        k=max(1,int(np.ceil(.05*len(l)))); cvar=float(np.mean(np.sort(l)[-k:])); return (1-risk_lambda)*mean+risk_lambda*cvar
    raise ValueError(mode)


def risk_object_batch(loss_matrix,mode="mean_cvar",risk_lambda=.2):
    l=np.asarray(loss_matrix,float)
    if l.ndim==1: l=l[:,None]
    mean=np.mean(l,axis=0)
    if mode=="mean": return mean
    if mode=="mean_sd":
        sd=np.std(l,axis=0,ddof=1) if l.shape[0]>1 else np.zeros(l.shape[1])
        return mean+risk_lambda*sd
    if mode=="mean_cvar":
        k=max(1,int(np.ceil(.05*l.shape[0])))
        tail=np.partition(l,l.shape[0]-k,axis=0)[-k:,:]
        cvar=np.mean(tail,axis=0)
        return (1-risk_lambda)*mean+risk_lambda*cvar
    raise ValueError(mode)


def decode_x_batch(X,joint_p=True,fixed_p=None):
    X=np.asarray(X,float)
    if X.ndim==1: X=X.reshape(1,-1)
    N=np.power(10.0,X[:,0]); D=np.power(10.0,X[:,1]); Q=X[:,2]
    if joint_p:
        logits=np.concatenate([X[:,3:],np.zeros((X.shape[0],1),float)],axis=1)
        logits=logits-logits.max(axis=1,keepdims=True); e=np.exp(logits); P=e/e.sum(axis=1,keepdims=True)
    else:
        P=np.repeat(np.asarray(fixed_p,float).reshape(1,-1),X.shape[0],axis=0)
    return N,D,Q,P


def make_objective(ens,budget,Q0,Lctx,cost_kind,bounds,joint_p=True,fixed_p=None,risk_mode="mean_cvar",risk_lambda=.2,penalty=1000):
    def f(x):
        N,D,Q,p=decode_x(x,joint_p,fixed_p); losses=ens.predict_samples(N,D,Q,p); val=risk_object(losses,risk_mode,risk_lambda)
        c,*_=total_cost(N,D,Q,Q0,Lctx,cost_kind)
        viol=max(c/budget-1.0,0.0)
        return val+penalty*viol*viol
    def cons(x):
        N,D,Q,p=decode_x(x,joint_p,fixed_p); c,*_=total_cost(N,D,Q,Q0,Lctx,cost_kind); return budget-c
    return f,cons


def make_vectorized_objective(ens,budget,Q0,Lctx,cost_kind,joint_p=True,fixed_p=None,risk_mode="mean_cvar",risk_lambda=.2,penalty=1000):
    """Population-batched DE objective; input candidates are rows."""
    def f_batch(Xrows):
        N,D,Q,P=decode_x_batch(Xrows,joint_p,fixed_p)
        losses=ens.predict_matrix(N,D,Q,P)
        val=risk_object_batch(losses,risk_mode,risk_lambda)
        train=6.0*N*D
        qcost=D*np.maximum(quality_cost_g(Q,cost_kind)-quality_cost_g(Q0,cost_kind),0.0)
        attn=ETA*N*D*Lctx
        c=train+qcost+attn
        viol=np.maximum(c/budget-1.0,0.0)
        return np.asarray(val+penalty*viol*viol,float)
    return f_batch


def scipy_bounds(spec:BoundsSpec,joint_p=True):
    b=[spec.logN,spec.logD,spec.Q]
    if joint_p:b += [spec.logit]*16
    return b


def random_x(bounds,rng): return np.array([rng.uniform(a,b) for a,b in bounds])


def solve_slsqp_multistart(ens,budget,Q0,Lctx,cost_kind,spec,joint_p=True,fixed_p=None,risk_mode="mean",risk_lambda=.2,n_starts=30,seed=42,x0=None,show_progress=True):
    bounds=scipy_bounds(spec,joint_p); obj,cons=make_objective(ens,budget,Q0,Lctx,cost_kind,spec,joint_p,fixed_p,risk_mode,risk_lambda)
    rng=np.random.default_rng(seed); starts=[]
    if x0 is not None: starts.append(np.asarray(x0,float))
    starts += [random_x(bounds,rng) for _ in range(n_starts)]
    best=None; t=time.time()
    iterator = tqdm(starts, desc="Q3 SLSQP multi-start", unit="start", leave=False, disable=not show_progress)
    for idx,s in enumerate(iterator,1):
        iter_bar=tqdm(total=2000,desc=f"Q3 SLSQP local {idx}/{len(starts)}",unit="iter",leave=False,disable=not show_progress)
        def local_callback(xk):
            iter_bar.update(1)
        try:
            r=minimize(obj,s,method="SLSQP",bounds=bounds,constraints=[{"type":"ineq","fun":cons}],callback=local_callback,options={"ftol":1e-9,"maxiter":2000,"disp":False})
        finally:
            iter_bar.close()
        iterator.set_postfix(last_iter=int(getattr(r,"nit",0)),best=(f"{best.fun:.4g}" if best is not None else "n/a"))
        if r.success and cons(r.x)>=-max(1,budget)*1e-8 and (best is None or r.fun<best.fun): best=r
    return best,time.time()-t


def solve_de(ens,budget,Q0,Lctx,cost_kind,spec,joint_p=True,fixed_p=None,risk_mode="mean",risk_lambda=.2,seed=42,maxiter=500,popsize=20,show_progress=True):
    bounds=scipy_bounds(spec,joint_p)
    obj_batch=make_vectorized_objective(ens,budget,Q0,Lctx,cost_kind,joint_p,fixed_p,risk_mode,risk_lambda)
    t=time.time(); ndim=len(bounds); pop_n=max(5,popsize*ndim)
    # SciPy vectorized DE calls the objective with shape (ndim, population).
    # Updating the bar in the batch objective means the initial population is
    # visible as the first unit instead of leaving a higher-level bar at 0.
    bar=tqdm(total=maxiter+1,desc="Q3 differential evolution",unit="population",leave=False,disable=not show_progress)
    state={"calls":0}
    bar.set_postfix_str(f"initial population: {pop_n} candidates")
    def scipy_vec(X):
        vals=obj_batch(np.asarray(X,float).T)
        state["calls"]+=1
        if state["calls"]<=bar.total:
            bar.update(1)
        if state["calls"]==1:
            bar.set_postfix_str(f"generation batches; {pop_n} candidates/batch")
        return vals
    def callback(xk,convergence):
        bar.set_postfix(conv=f"{convergence:.2e}",batch=f"{pop_n}")
        return False
    try:
        r=differential_evolution(scipy_vec,bounds,seed=seed,popsize=popsize,maxiter=maxiter,tol=1e-7,polish=False,workers=1,updating="deferred",vectorized=True,callback=callback)
    finally:
        bar.close()
    return r,time.time()-t


def solve_hybrid(ens,budget,Q0,Lctx,cost_kind,spec,joint_p=True,fixed_p=None,risk_mode="mean_cvar",risk_lambda=.2,seed=42,de_maxiter=500,de_popsize=20,x0=None,fast=False,show_progress=True):
    if x0 is None:
        de,td=solve_de(ens,budget,Q0,Lctx,cost_kind,spec,joint_p,fixed_p,risk_mode,risk_lambda,seed,maxiter=(2 if fast else de_maxiter),popsize=(2 if fast else de_popsize),show_progress=show_progress)
        start=de.x
    else:
        td=0.0; start=x0
    sl,ts=solve_slsqp_multistart(ens,budget,Q0,Lctx,cost_kind,spec,joint_p,fixed_p,risk_mode,risk_lambda,n_starts=(1 if fast else 5),seed=seed,x0=start,show_progress=show_progress)
    if sl is not None: return sl,td+ts
    # fallback to DE result if no local feasible solution
    if x0 is None:return de,td
    de,td2=solve_de(ens,budget,Q0,Lctx,cost_kind,spec,joint_p,fixed_p,risk_mode,risk_lambda,seed,maxiter=(2 if fast else de_maxiter),popsize=(2 if fast else de_popsize),show_progress=show_progress)
    return de,td+ts+td2


def result_row(res,elapsed,ens,budget,Q0,Lctx,cost_kind,joint_p,fixed_p,risk_mode,risk_lambda,algorithm):
    N,D,Q,p=decode_x(res.x,joint_p,fixed_p); losses=ens.predict_samples(N,D,Q,p); c,ct,cq,ca=total_cost(N,D,Q,Q0,Lctx,cost_kind)
    row={"algorithm":algorithm,"risk_mode":risk_mode,"risk_lambda":risk_lambda,"cost_kind":cost_kind,"budget":budget,"L_ctx":Lctx,
         "N_opt":N,"D_opt":D,"Q_opt":Q,"expected_loss":float(np.mean(losses)),"loss_sd":float(np.std(losses,ddof=1) if len(losses)>1 else 0),
         "loss_p95":float(np.quantile(losses,.95)),"cvar95":float(np.mean(np.sort(losses)[-max(1,int(np.ceil(.05*len(losses)))):])),
         "objective":float(res.fun),"C_total":c,"C_train":ct,"C_quality":cq,"C_attn":ca,"budget_utilization":c/budget,
         "r_train":ct/budget,"r_quality":cq/budget,"r_attn":ca/budget,"mixture_entropy":composition_entropy(p),"runtime_sec":elapsed,
         "success":bool(getattr(res,"success",True)),"message":str(getattr(res,"message",""))}
    for i,v in enumerate(p):row[f"p_{i:02d}"]=v
    return row


def finite_difference_kkt(ens,row,Q0,cost_kind,eps=1e-4):
    N,D,Q=row["N_opt"],row["D_opt"],row["Q_opt"]; p=np.array([row[f"p_{i:02d}"] for i in range(17)]); Lctx=row["L_ctx"]
    def L(n,d,q):return ens.mean(n,d,q,p)
    dN=N*eps; dD=D*eps; dQ=max(1e-5,eps)
    LN=(L(N+dN,D,Q)-L(N-dN,D,Q))/(2*dN); LD=(L(N,D+dD,Q)-L(N,D-dD,Q))/(2*dD); LQ=(L(N,D,min(1,Q+dQ))-L(N,D,max(Q0,Q-dQ)))/(min(1,Q+dQ)-max(Q0,Q-dQ)+1e-30)
    CN=D*(6+ETA*Lctx); CD=N*(6+ETA*Lctx)+max(float(quality_cost_g(Q,cost_kind)-quality_cost_g(Q0,cost_kind)),0); CQ=D*quality_cost_g_prime(Q,cost_kind) if Q>Q0+1e-8 else np.nan
    ratios={"mb_N":-LN/CN,"mb_D":-LD/CD,"mb_Q":(-LQ/CQ if np.isfinite(CQ) else np.nan)}
    vals=[v for v in ratios.values() if np.isfinite(v)]; ratios["kkt_ratio_spread"]=float(max(vals)-min(vals)) if vals else np.nan
    return ratios
