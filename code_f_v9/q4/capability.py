from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm.auto import trange

BENCH_COLS=["ifeval","bbh","math_lvl5","gpqa","musr","mmlu_pro"]


@dataclass
class CapabilityBundle:
    scaler: StandardScaler
    pca: PCA
    sign: float
    bench_cols: list[str]
    def transform(self,df:pd.DataFrame):
        X=df[self.bench_cols].to_numpy(float); z=self.scaler.transform(X); pc=self.pca.transform(z)[:,0]*self.sign; return pc


def fit_capability(df:pd.DataFrame):
    x=df.dropna(subset=BENCH_COLS).copy(); X=x[BENCH_COLS].to_numpy(float)
    scaler=StandardScaler().fit(X); z=scaler.transform(X); pca=PCA().fit(z); pc=pca.transform(z)[:,0]
    raw=X.mean(axis=1); sign=1.0 if np.corrcoef(pc,raw)[0,1]>=0 else -1.0; pc*=sign
    bundle=CapabilityBundle(scaler,pca,sign,BENCH_COLS)
    scores=x[[c for c in ["model","model_key","submission_date","params_B","model_type_bucket","hub_license","epoch_open_weights","benchmark_complete"] if c in x.columns]].copy()
    scores["cap_raw_mean"]=raw; scores["cap_z_mean"]=z.mean(axis=1); scores["cap_pca"]=pc; scores["pc2"]=pca.transform(z)[:,1] if pca.n_components_>1 else np.nan
    load=pd.DataFrame({"benchmark":BENCH_COLS,"pc1_loading":pca.components_[0]*sign})
    ev=pd.DataFrame({"component":np.arange(1,len(pca.explained_variance_ratio_)+1),"explained_variance_ratio":pca.explained_variance_ratio_,"cumulative":np.cumsum(pca.explained_variance_ratio_)})
    return bundle,scores,load,ev


def capability_stability(df,bundle,n_boot=1000,seed=42):
    x=df.dropna(subset=BENCH_COLS).copy(); X=x[BENCH_COLS].to_numpy(float); base=bundle.transform(x); rng=np.random.default_rng(seed); rows=[]
    n=len(x)
    for b in trange(n_boot, desc="Q4 capability bootstrap", unit="rep", leave=False):
        idx=rng.integers(0,n,size=n); sc=StandardScaler().fit(X[idx]); z=sc.transform(X[idx]); p=PCA().fit(z); # orient using bootstrap raw mean
        train_pc=p.transform(z)[:,0]; sg=1.0 if np.corrcoef(train_pc,X[idx].mean(axis=1))[0,1]>=0 else -1.0
        zorig=sc.transform(X); pred=p.transform(zorig)[:,0]*sg; rho=spearmanr(base,pred).statistic
        zmean=zorig.mean(axis=1); base_z=bundle.scaler.transform(X).mean(axis=1); rho_z=spearmanr(base_z,zmean).statistic
        rows.append({"bootstrap":b,"pca_rank_spearman":float(rho),"zmean_rank_spearman":float(rho_z),"raw_mean_rank_spearman":1.0,"pc1_explained":float(p.explained_variance_ratio_[0])})
    # leave-one-benchmark sensitivity
    leave=[]
    for drop in BENCH_COLS:
        cols=[c for c in BENCH_COLS if c!=drop]; sc=StandardScaler().fit(x[cols]); z=sc.transform(x[cols]); p=PCA().fit(z); pc=p.transform(z)[:,0]; sg=1.0 if np.corrcoef(pc,x[cols].mean(axis=1))[0,1]>=0 else -1.0; rho=spearmanr(base,pc*sg).statistic
        leave.append({"dropped_benchmark":drop,"rank_spearman":float(rho)})
    return pd.DataFrame(rows),pd.DataFrame(leave)


def select_task_metric(task_group,metric):
    m=str(metric).lower(); g=str(task_group).upper()
    priorities={
        "IFEVAL":["prompt_level_strict_acc,none","prompt_level_strict_acc"],
        "BBH":["acc_norm,none","acc,none"],
        "MATH":["exact_match,none","exact_match"],
        "GPQA":["acc_norm,none","acc,none"],
        "MUSR":["acc_norm,none","acc,none"],
        "MMLU_PRO":["acc,none","acc_norm,none"],
    }
    ps=priorities.get(g,[])
    return ps.index(m) if m in ps else 999


def aggregate_c8(tasks:pd.DataFrame):
    x=tasks.copy(); x=x[x.task_group.isin(["IFEval","BBH","MATH","GPQA","MUSR","MMLU_PRO"])].copy()
    # Detailed result JSONs can contain uncertainty/alias fields such as stderr,
    # acc_stderr or *_stderr. They are not task scores and must never be selected
    # merely because the canonical metric is absent. Retain score-like metrics only.
    metric_lower=x["metric"].astype(str).str.lower()
    bad=metric_lower.str.contains("stderr|std_err|standard_error|alias|samples|n-samples|n_samples",regex=True,na=False)
    score_like=metric_lower.str.contains("acc|exact_match|f1|score|pass|strict",regex=True,na=False)
    x=x[(~bad)&score_like].copy()
    x["metric_priority"]=[select_task_metric(g,m) for g,m in zip(x.task_group,x.metric)]
    # Prefer benchmark-specific canonical metrics per task. If none exists, use a
    # deterministic score-metric preference rather than arbitrary JSON field order.
    fallback_order={
        "acc_norm,none":10,"acc,none":11,"exact_match,none":12,"prompt_level_strict_acc,none":13,
        "acc_norm":20,"acc":21,"exact_match":22,"f1":23,"score":24,
    }
    fallback=x["metric"].astype(str).str.lower().map(fallback_order).fillna(100).astype(int)
    x["metric_priority"]=np.where(x.metric_priority<999,x.metric_priority,fallback)
    best=x.groupby(["model_key","task"])["metric_priority"].transform("min"); x=x[x.metric_priority.eq(best)].copy()
    # Multiple same-priority metric rows are averaged, then subtask macro-average within benchmark.
    task=x.groupby(["model_key","model_name","task_group","task"],as_index=False).agg(score=("value","mean"),n_samples=("n_samples_effective","max"))
    # Convert fraction-like values to percentage scale for comparability.
    task["score_pct"]=np.where(task.score.between(-1.01,1.01),task.score*100,task.score)
    group=task.groupby(["model_key","model_name","task_group"],as_index=False).agg(group_score=("score_pct","mean"),n_subtasks=("task","nunique"))
    # Standardize benchmark-group scores globally, then specialization = SD across groups for each model.
    group["group_z"]=group.groupby("task_group")["group_score"].transform(lambda s:(s-s.mean())/(s.std(ddof=1)+1e-12))
    spec=group.groupby(["model_key","model_name"],as_index=False).agg(c8_mean_z=("group_z","mean"),c8_specialization_sd=("group_z","std"),c8_groups=("task_group","nunique"))
    return task,group,spec


def pareto_flags(X, show_progress=False, desc="Q4 Pareto frontier"):
    X=np.asarray(X,float); n=len(X); eff=np.ones(n,dtype=bool)
    iterator=trange(n, desc=desc, unit="model", leave=False, disable=not show_progress)
    for i in iterator:
        if not eff[i]: continue
        dominated=np.all(X>=X[i],axis=1)&np.any(X>X[i],axis=1)
        dominated[i]=False
        if np.any(dominated): eff[i]=False
    return eff
