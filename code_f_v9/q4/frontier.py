from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.preprocessing import SplineTransformer,OneHotEncoder,StandardScaler
from common.metrics import pinball_loss

try:
    import cvxpy as cp
except Exception:
    cp=None


class QuantileGAM:
    def __init__(self,tau=.95,n_knots_time=6,n_knots_compute=5,lam_time=1.0,lam_compute=1.0):
        self.tau=tau;self.n_knots_time=n_knots_time;self.n_knots_compute=n_knots_compute;self.lam_time=lam_time;self.lam_compute=lam_compute
    def fit(self,df,y,use_compute=True):
        self.use_compute=use_compute
        self.tsp=SplineTransformer(n_knots=max(2,min(self.n_knots_time,df.time_months.nunique())),degree=3,include_bias=False,extrapolation="linear").fit(df[["time_months"]]); T=self.tsp.transform(df[["time_months"]]); parts=[T];self.slices={"time":slice(0,T.shape[1])};start=T.shape[1]
        if use_compute:
            self.csp=SplineTransformer(n_knots=max(2,min(self.n_knots_compute,df.logC.nunique())),degree=3,include_bias=False,extrapolation="linear").fit(df[["logC"]]); C=self.csp.transform(df[["logC"]]);parts.append(C);self.slices["compute"]=slice(start,start+C.shape[1]);start+=C.shape[1]
        try:self.oh=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        except TypeError:self.oh=OneHotEncoder(handle_unknown="ignore",sparse=False)
        O=self.oh.fit_transform(df[["model_type_bucket"]].astype(str));parts.append(O);self.slices["type"]=slice(start,start+O.shape[1]);X=np.c_[*parts];self.scaler=StandardScaler().fit(X);X=self.scaler.transform(X);y=np.asarray(y,float)
        if cp is None:
            # Fallback: sklearn quantile regression on spline design, without explicit roughness penalty.
            self.beta_model=QuantileRegressor(quantile=self.tau,alpha=1e-4,solver="highs").fit(X,y);self.beta_=self.beta_model.coef_;self.intercept_=self.beta_model.intercept_;return self
        beta=cp.Variable(X.shape[1]);b=cp.Variable();u=y-(b+X@beta);loss=cp.sum(self.tau*cp.pos(u)+(1-self.tau)*cp.pos(-u));pen=0
        for key,lam in [("time",self.lam_time),("compute",self.lam_compute)]:
            if key in self.slices:
                sl=self.slices[key]; inds=np.arange(sl.start,sl.stop); D2=np.diff(np.eye(len(inds)),n=2,axis=0); pen+=lam*cp.sum_squares(D2@beta[inds])
        prob=cp.Problem(cp.Minimize(loss+pen));solved=False
        for solver in ["CLARABEL","SCS"]:
            try:prob.solve(solver=solver,verbose=False);solved=prob.status in {"optimal","optimal_inaccurate"}
            except Exception:pass
            if solved:break
        if not solved:raise RuntimeError(f"Quantile GAM solve failed: {prob.status}")
        self.beta_=np.asarray(beta.value,float);self.intercept_=float(b.value);self.beta_model=None;return self
    def _design(self,df):
        parts=[self.tsp.transform(df[["time_months"]])]
        if self.use_compute:parts.append(self.csp.transform(df[["logC"]]))
        parts.append(self.oh.transform(df[["model_type_bucket"]].astype(str)));return self.scaler.transform(np.c_[*parts])
    def predict(self,df):
        X=self._design(df);return self.intercept_+X@self.beta_



def license_bucket(value):
    s=str(value).strip().lower()
    if not s or s in {"nan","none","unknown"}:
        return "unknown"
    permissive=["apache","mit","bsd","cc-by","openrail","mpl","isc"]
    restricted=["research","non-commercial","noncommercial","nc","llama","gemma","qwen","community","custom","restricted"]
    if any(k in s for k in permissive):
        return "permissive_or_open"
    if any(k in s for k in restricted):
        return "research_or_restricted"
    return "other_or_unclear"

def is_open_weights(row):
    # Prefer explicit Epoch boolean; fallback to enhanced leaderboard field / permissive-license heuristic.
    for c in ["open_weights_bool_final","open_weights_bool","epoch_open_weights"]:
        if c in row and pd.notna(row[c]):
            v=str(row[c]).strip().lower()
            if v in {"true","1","yes","y"}:return True
            if v in {"false","0","no","n"}:return False
    lic=str(row.get("hub_license","" )).lower()
    return any(k in lic for k in ["apache","mit","bsd","cc-by","openrail"])


def prepare_frontier_panel(model_panel:pd.DataFrame):
    x=model_panel.copy();x["submission_date"]=pd.to_datetime(x.submission_date,errors="coerce");x=x.dropna(subset=["submission_date","cap_pca"]);x["license_bucket"]=x.get("hub_license",pd.Series("",index=x.index)).map(license_bucket);x["open_weights"]=x.apply(is_open_weights,axis=1);x=x[x.open_weights].copy();t0=x.submission_date.min();x["time_months"]=(x.submission_date-t0).dt.days/30.4375;x["logC"]=np.log10(pd.to_numeric(x.C,errors="coerce").where(pd.to_numeric(x.C,errors="coerce")>0));return x,t0


def _bool_series(s:pd.Series):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes","y"})


def compute_frontier_series(epoch:pd.DataFrame,quant=.9):
    x=epoch.copy(); x["publication_date"]=pd.to_datetime(x.publication_date,errors="coerce")
    if "open_weights_bool" not in x.columns:
        raise KeyError("q4_epoch_models.csv must contain open_weights_bool")
    x=x[_bool_series(x.open_weights_bool)].copy()
    x["training_compute_flop"]=pd.to_numeric(x.training_compute_flop,errors="coerce")
    x=x.dropna(subset=["publication_date","training_compute_flop"]);x=x[x.training_compute_flop>0].copy();x["quarter"]=x.publication_date.dt.to_period("Q").dt.start_time
    q=x.groupby("quarter").training_compute_flop.quantile(quant).reset_index(name="compute_q");q["logC_q"]=np.log10(q.compute_q);return q.sort_values("quarter").reset_index(drop=True)


def fit_compute_growth(series,recent_months=24):
    x=series.sort_values("quarter").dropna(subset=["quarter","logC_q"]).copy()
    if len(x)<2:
        raise ValueError("At least two quarterly compute-frontier observations are required")
    last=x.quarter.max();use=x[x.quarter>=last-pd.DateOffset(months=recent_months)].copy()
    if len(use)<2:
        use=x.tail(min(len(x),8)).copy()
    t=(use.quarter-use.quarter.min()).dt.days/30.4375
    if np.unique(t).size<2:
        use=x.copy(); t=(use.quarter-use.quarter.min()).dt.days/30.4375
    reg=LinearRegression().fit(t.to_numpy().reshape(-1,1),use.logC_q.to_numpy(float))
    return float(reg.coef_[0]),float(reg.intercept_),use.quarter.min(),last,float(x.iloc[-1].logC_q)


def rolling_empirical_forecast(train,target_date,window_months=12,tau=.95):
    lo=target_date-pd.DateOffset(months=window_months);use=train[(train.submission_date>=lo)&(train.submission_date<=train.submission_date.max())];return float(use.cap_pca.quantile(tau)) if len(use) else np.nan

# -----------------------------------------------------------------------------
# Deep quantile frontier models (F-D / F-E / F-F)
# -----------------------------------------------------------------------------
import copy
import torch
from torch import nn


def _torch_pinball(pred, target, tau):
    u = target - pred
    return torch.mean(torch.maximum(tau * u, (tau - 1.0) * u))


def _type_categories(df):
    vals = sorted(df["model_type_bucket"].astype(str).fillna("unknown").unique().tolist())
    return vals if vals else ["unknown"]


def _type_index(series, mapping):
    return np.array([mapping.get(str(v), 0) for v in series], dtype=np.int64)


class _PlainQuantileMLP(nn.Module):
    def __init__(self, n_numeric, n_types, hidden=(32, 16), emb_dim=4):
        super().__init__()
        self.emb = nn.Embedding(max(1, n_types), emb_dim)
        dims = [n_numeric + emb_dim, *hidden, 1]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.SiLU(), nn.Dropout(0.05)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, xnum, xtype):
        e = self.emb(xtype)
        return self.net(torch.cat([xnum, e], dim=1)).squeeze(1)


class _PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, init_raw=-2.0):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.full((out_features, in_features), float(init_raw)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.nn.functional.linear(x, torch.nn.functional.softplus(self.raw_weight), self.bias)


class _PositiveBranch(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.l1 = _PositiveLinear(1, hidden)
        self.l2 = _PositiveLinear(hidden, hidden)
        self.out = _PositiveLinear(hidden, 1)
        self.act = nn.Softplus()

    def raw(self, x):
        return self.out(self.act(self.l2(self.act(self.l1(x))))).squeeze(1)

    def forward(self, x):
        # Anchor at x=0. For x>=0 this remains non-negative and monotone.
        z0 = torch.zeros_like(x)
        return self.raw(x) - self.raw(z0)


class _MonotonicQuantileNet(nn.Module):
    def __init__(self, n_types, use_compute=True, interaction=False, hidden=8, emb_dim=1):
        super().__init__()
        self.use_compute = use_compute
        self.interaction = interaction and use_compute
        self.time_branch = _PositiveBranch(hidden)
        self.compute_branch = _PositiveBranch(hidden) if use_compute else None
        # Type enters as an additive learned bias; it does not affect monotonic derivatives.
        self.type_bias = nn.Embedding(max(1, n_types), emb_dim)
        self.type_proj = nn.Linear(emb_dim, 1, bias=False)
        self.intercept = nn.Parameter(torch.zeros(1))
        self.raw_gamma = nn.Parameter(torch.tensor(-3.0)) if self.interaction else None

    def forward(self, t, c, xtype):
        ft = self.time_branch(t)
        fc = self.compute_branch(c) if self.use_compute else torch.zeros_like(ft)
        y = self.intercept + ft + fc + self.type_proj(self.type_bias(xtype)).squeeze(1)
        if self.interaction:
            # Lightweight positive t x C interaction.  Softplus keeps both factors
            # positive even under modest extrapolation below the observed minimum,
            # so gamma>=0 preserves monotonicity in both time and compute globally.
            gamma = torch.nn.functional.softplus(self.raw_gamma)
            y = y + gamma * torch.nn.functional.softplus(ft) * torch.nn.functional.softplus(fc)
        return y


class DeepQuantileRegressor:
    """F-D: unconstrained deep quantile MLP on time, compute and model type."""
    def __init__(self, tau=.95, lr=1e-3, max_epochs=1200, patience=120, seed=42):
        self.tau=tau; self.lr=lr; self.max_epochs=max_epochs; self.patience=patience; self.seed=seed

    def _prepare_fit(self, df, use_compute):
        self.use_compute = bool(use_compute)
        self.numeric_cols = ["time_months"] + (["logC"] if self.use_compute else [])
        X = df[self.numeric_cols].to_numpy(float)
        self.mean_ = X.mean(axis=0); self.std_ = X.std(axis=0); self.std_[self.std_ < 1e-9] = 1.0
        self.types_ = _type_categories(df); self.type_map_ = {v:i for i,v in enumerate(self.types_)}
        return (X-self.mean_)/self.std_, _type_index(df["model_type_bucket"].astype(str), self.type_map_)

    def fit(self, df, y, use_compute=True, show_progress=False):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X, T = self._prepare_fit(df, use_compute); y=np.asarray(y,float)
        order=np.argsort(pd.to_datetime(df["submission_date"],errors="coerce").fillna(pd.Timestamp("1970-01-01")).to_numpy())
        n=len(order); nval=max(3,int(round(.2*n))) if n>=20 else max(1,int(round(.15*n)))
        tr=order[:-nval] if n-nval>=5 else order; va=order[-nval:] if n-nval>=5 else order
        self.model_=_PlainQuantileMLP(X.shape[1],len(self.types_)); opt=torch.optim.AdamW(self.model_.parameters(),lr=self.lr,weight_decay=1e-4)
        xt=torch.tensor(X,dtype=torch.float32); tt=torch.tensor(T,dtype=torch.long); yt=torch.tensor(y,dtype=torch.float32)
        best=None; best_loss=np.inf; stale=0
        for epoch in range(self.max_epochs):
            self.model_.train(); opt.zero_grad(); pred=self.model_(xt[tr],tt[tr]); loss=_torch_pinball(pred,yt[tr],self.tau); loss.backward(); nn.utils.clip_grad_norm_(self.model_.parameters(),1.0); opt.step()
            self.model_.eval()
            with torch.no_grad(): vl=float(_torch_pinball(self.model_(xt[va],tt[va]),yt[va],self.tau))
            if vl < best_loss-1e-6: best_loss=vl; best=copy.deepcopy(self.model_.state_dict()); stale=0
            else: stale+=1
            if stale>=self.patience: break
        if best is not None:self.model_.load_state_dict(best)
        self.best_val_loss_=best_loss; self.epochs_=epoch+1; return self

    def predict(self, df):
        X=df[self.numeric_cols].to_numpy(float); X=(X-self.mean_)/self.std_; T=_type_index(df["model_type_bucket"].astype(str),self.type_map_)
        self.model_.eval()
        with torch.no_grad(): return self.model_(torch.tensor(X,dtype=torch.float32),torch.tensor(T,dtype=torch.long)).cpu().numpy()


class MonotonicDeepQuantileRegressor:
    """F-E/F-F: monotone deep quantile model, optionally with lightweight positive t x C interaction."""
    def __init__(self, tau=.95, interaction=False, lr=7e-4, max_epochs=1500, patience=150, seed=42):
        self.tau=tau; self.interaction=interaction; self.lr=lr; self.max_epochs=max_epochs; self.patience=patience; self.seed=seed

    def _prepare_fit(self, df, use_compute):
        self.use_compute=bool(use_compute)
        self.tmin_=float(df.time_months.min()); self.tscale_=max(float(df.time_months.max()-self.tmin_),1e-6)
        if self.use_compute:
            self.cmin_=float(df.logC.min()); self.cscale_=max(float(df.logC.max()-self.cmin_),1e-6)
        else:self.cmin_=0.0; self.cscale_=1.0
        self.types_=_type_categories(df); self.type_map_={v:i for i,v in enumerate(self.types_)}
        t=((df.time_months.to_numpy(float)-self.tmin_)/self.tscale_).reshape(-1,1)
        c=((df.logC.to_numpy(float)-self.cmin_)/self.cscale_).reshape(-1,1) if self.use_compute else np.zeros_like(t)
        typ=_type_index(df.model_type_bucket.astype(str),self.type_map_)
        return t,c,typ

    def fit(self, df, y, use_compute=True, show_progress=False):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        t,c,typ=self._prepare_fit(df,use_compute); y=np.asarray(y,float)
        order=np.argsort(pd.to_datetime(df["submission_date"],errors="coerce").fillna(pd.Timestamp("1970-01-01")).to_numpy())
        n=len(order); nval=max(3,int(round(.2*n))) if n>=20 else max(1,int(round(.15*n)))
        tr=order[:-nval] if n-nval>=5 else order; va=order[-nval:] if n-nval>=5 else order
        self.model_=_MonotonicQuantileNet(len(self.types_),use_compute=self.use_compute,interaction=self.interaction)
        # Initialize intercept close to the empirical target quantile.
        with torch.no_grad(): self.model_.intercept.fill_(float(np.quantile(y,self.tau)))
        opt=torch.optim.AdamW(self.model_.parameters(),lr=self.lr,weight_decay=3e-4)
        tx=torch.tensor(t,dtype=torch.float32); cx=torch.tensor(c,dtype=torch.float32); kx=torch.tensor(typ,dtype=torch.long); yx=torch.tensor(y,dtype=torch.float32)
        best=None; best_loss=np.inf; stale=0
        for epoch in range(self.max_epochs):
            self.model_.train(); opt.zero_grad(); pred=self.model_(tx[tr],cx[tr],kx[tr]); loss=_torch_pinball(pred,yx[tr],self.tau)
            # Mild L2 penalty on interaction coefficient prevents F-F from becoming a black box.
            if self.interaction: loss=loss+1e-3*torch.nn.functional.softplus(self.model_.raw_gamma).pow(2)
            loss.backward(); nn.utils.clip_grad_norm_(self.model_.parameters(),1.0); opt.step()
            self.model_.eval()
            with torch.no_grad(): vl=float(_torch_pinball(self.model_(tx[va],cx[va],kx[va]),yx[va],self.tau))
            if vl < best_loss-1e-6: best_loss=vl; best=copy.deepcopy(self.model_.state_dict()); stale=0
            else: stale+=1
            if stale>=self.patience: break
        if best is not None:self.model_.load_state_dict(best)
        self.best_val_loss_=best_loss; self.epochs_=epoch+1; return self

    def _inputs(self,df):
        t=((df.time_months.to_numpy(float)-self.tmin_)/self.tscale_).reshape(-1,1)
        c=((df.logC.to_numpy(float)-self.cmin_)/self.cscale_).reshape(-1,1) if self.use_compute else np.zeros_like(t)
        typ=_type_index(df.model_type_bucket.astype(str),self.type_map_)
        return t,c,typ

    def predict(self,df):
        t,c,typ=self._inputs(df); self.model_.eval()
        with torch.no_grad(): return self.model_(torch.tensor(t,dtype=torch.float32),torch.tensor(c,dtype=torch.float32),torch.tensor(typ,dtype=torch.long)).cpu().numpy()

    def monotonicity_check(self,df,n_grid=80):
        """Return finite-difference violation counts over representative time/compute grids."""
        typ=str(df.model_type_bucket.mode().iloc[0])
        tmin,tobs=float(df.time_months.min()),float(df.time_months.max()); tmax=tobs+0.25*max(tobs-tmin,1e-6)
        if self.use_compute:
            cmin,cobs=float(df.logC.min()),float(df.logC.max()); cmax=cobs+0.25*max(cobs-cmin,1e-6)
        else:cmin=cmax=0.0
        rows=[]
        for cval in ([cmin,(cmin+cmax)/2,cmax] if self.use_compute else [0.0]):
            g=pd.DataFrame({"time_months":np.linspace(tmin,tmax,n_grid),"model_type_bucket":typ})
            if self.use_compute:g["logC"]=cval
            p=self.predict(g); rows.append({"axis":"time","conditioning":cval,"violations":int(np.sum(np.diff(p)<-1e-7)),"min_diff":float(np.min(np.diff(p)))})
        if self.use_compute:
            for tval in [tmin,(tmin+tmax)/2,tmax]:
                g=pd.DataFrame({"time_months":tval,"logC":np.linspace(cmin,cmax,n_grid),"model_type_bucket":typ}); p=self.predict(g)
                rows.append({"axis":"compute","conditioning":tval,"violations":int(np.sum(np.diff(p)<-1e-7)),"min_diff":float(np.min(np.diff(p)))})
        return pd.DataFrame(rows)
