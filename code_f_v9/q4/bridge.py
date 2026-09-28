from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm, trange

from common.metrics import weighted_regression_metrics

try:
    import cvxpy as cp
except Exception:
    cp=None


class ISplineBasis:
    def __init__(self,degree=3,quantiles=(.1,.3,.5,.7,.9)):
        self.degree=degree; self.quantiles=quantiles
    def fit(self,x):
        x=np.asarray(x,float); x=x[np.isfinite(x)]; self.xmin=float(np.min(x)); self.xmax=float(np.max(x))
        if self.xmax<=self.xmin:self.xmax=self.xmin+1e-6
        internal=np.unique(np.quantile(x,self.quantiles)); internal=internal[(internal>self.xmin)&(internal<self.xmax)]
        self.knots=np.r_[[self.xmin]*(self.degree+1),internal,[self.xmax]*(self.degree+1)]
        self.n_basis=len(self.knots)-self.degree-1
        self._anti=[]; self._factor=[]
        eye=np.eye(self.n_basis)
        for i in range(self.n_basis):
            bs=BSpline(self.knots,eye[i],self.degree,extrapolate=True); anti=bs.antiderivative(); den=self.knots[i+self.degree+1]-self.knots[i]; factor=(self.degree+1)/den if den>0 else 0.0
            self._anti.append(anti); self._factor.append(factor)
        return self
    def transform(self,x):
        v=np.clip(np.asarray(x,float),self.xmin,self.xmax); out=[]
        for anti,f in zip(self._anti,self._factor): out.append(f*(anti(v)-anti(self.xmin)))
        I=np.column_stack(out); return np.clip(I,0.0,1.0)


class WeightedISplineRegressor:
    def __init__(self,degree=3,quantiles=(.1,.3,.5,.7,.9),lam=1e-2):
        self.degree=degree; self.quantiles=quantiles; self.lam=float(lam); self.basis=ISplineBasis(degree,quantiles)
    def fit(self,x,y,sample_weight=None):
        x=np.asarray(x,float); y=np.asarray(y,float); w=np.ones_like(y) if sample_weight is None else np.asarray(sample_weight,float)
        self.basis.fit(x); I=self.basis.transform(x)
        if cp is not None:
            a=cp.Variable(I.shape[1],nonneg=True); b=cp.Variable(); resid=y-(b-I@a); obj=cp.sum(cp.multiply(w,cp.square(resid)))+self.lam*cp.sum_squares(a); prob=cp.Problem(cp.Minimize(obj));
            solved=False
            for solver in ["OSQP","CLARABEL","SCS"]:
                try: prob.solve(solver=solver,warm_start=True,verbose=False); solved=prob.status in {"optimal","optimal_inaccurate"}
                except Exception: pass
                if solved: break
            if not solved: raise RuntimeError(f"I-spline cvxpy solve failed: {prob.status}")
            self.intercept_=float(b.value); self.coef_=np.asarray(a.value,float)
        else:
            def fun(v):
                b=v[0]; a=v[1:]; r=y-(b-I@a); return float(np.sum(w*r*r)+self.lam*np.sum(a*a))
            x0=np.r_[np.average(y,weights=w),np.ones(I.shape[1])*.01]; bounds=[(None,None)]+[(0,None)]*I.shape[1]; r=minimize(fun,x0,bounds=bounds,method="L-BFGS-B"); self.intercept_=float(r.x[0]); self.coef_=r.x[1:]
        return self
    def predict(self,x): return self.intercept_-self.basis.transform(np.asarray(x,float))@self.coef_


class WeightedSplineRegressor:
    def __init__(self,n_knots=6,degree=3,alpha=.1):self.n_knots=n_knots;self.degree=degree;self.alpha=alpha
    def fit(self,x,y,sample_weight=None):
        self.sp=SplineTransformer(n_knots=self.n_knots,degree=self.degree,include_bias=False,extrapolation="linear"); X=self.sp.fit_transform(np.asarray(x).reshape(-1,1)); self.model=Ridge(alpha=self.alpha).fit(X,y,sample_weight=sample_weight); return self
    def predict(self,x):return self.model.predict(self.sp.transform(np.asarray(x).reshape(-1,1)))


class LinearWeighted:
    def fit(self,x,y,sample_weight=None): self.model=LinearRegression().fit(np.asarray(x).reshape(-1,1),y,sample_weight=sample_weight); return self
    def predict(self,x): return self.model.predict(np.asarray(x).reshape(-1,1))


class IsotonicWeighted:
    def fit(self,x,y,sample_weight=None): self.model=IsotonicRegression(increasing=False,out_of_bounds="clip").fit(x,y,sample_weight=sample_weight); return self
    def predict(self,x): return self.model.predict(x)


def comparability_weights(bucket,omega=.5):
    s=pd.Series(bucket).astype(str).str.lower(); return np.where(s.eq("high"),1.0,np.where(s.eq("medium"),omega,0.25*omega))


def select_ispline_hyperparams(x,y,bucket,seed=42):
    x=np.asarray(x,float);y=np.asarray(y,float);b=np.asarray(bucket); high=np.where(pd.Series(b).astype(str).str.lower().eq("high"))[0]
    if len(high)<5:return .5,1e-2
    kf=KFold(n_splits=min(5,len(high)),shuffle=True,random_state=seed); best=None
    combos=[(omega,lam) for omega in [.25,.5,.75] for lam in [0,1e-4,1e-3,1e-2,1e-1,1,10]]
    for omega,lam in tqdm(combos, desc="Q4 I-spline hyperparameter CV", unit="combo", leave=False):
        w=comparability_weights(b,omega)
        errs=[]
        for trh,teh in kf.split(high):
            test=high[teh]; mask=np.ones(len(x),dtype=bool); mask[test]=False
            m=WeightedISplineRegressor(lam=lam).fit(x[mask],y[mask],w[mask]); p=m.predict(x[test]); errs.extend((p-y[test])**2)
        score=float(np.mean(errs))
        if best is None or score<best[0]:best=(score,omega,lam)
    return best[1],best[2]


def monotonic_violation(model,xmin,xmax,n=400):
    g=np.linspace(xmin,xmax,n); p=model.predict(g); return float(np.mean(np.diff(p)>1e-8))


def loocv_models(x,y,bucket,omega,lam):
    x=np.asarray(x,float);y=np.asarray(y,float);w=comparability_weights(bucket,omega); n=len(y); preds={k:np.full(n,np.nan) for k in ["linear","spline","isotonic","i_spline"]}
    for i in trange(n, desc="Q4 bridge LOOCV", unit="held-out", leave=False):
        mask=np.ones(n,dtype=bool);mask[i]=False
        mods={"linear":LinearWeighted(),"spline":WeightedSplineRegressor(),"isotonic":IsotonicWeighted(),"i_spline":WeightedISplineRegressor(lam=lam)}
        for name,m in mods.items():
            try:m.fit(x[mask],y[mask],w[mask]);preds[name][i]=m.predict([x[i]])[0]
            except Exception:pass
    return preds,w


def fit_all_models(x,y,bucket,omega,lam):
    w=comparability_weights(bucket,omega); mods={"linear":LinearWeighted(),"spline":WeightedSplineRegressor(),"isotonic":IsotonicWeighted(),"i_spline":WeightedISplineRegressor(lam=lam)}
    for m in mods.values():m.fit(x,y,w)
    return mods,w
