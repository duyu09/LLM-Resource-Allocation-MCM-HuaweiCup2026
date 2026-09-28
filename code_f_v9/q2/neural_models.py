from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import trange


def positive(raw):
    return F.softplus(raw)


class PositiveLinear(nn.Module):
    """Linear layer with non-negative weights, parameterized through softplus."""
    def __init__(self, in_features, out_features, bias=True, init_raw=-3.0):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        if init_raw is None:
            nn.init.xavier_uniform_(self.raw_weight)
        else:
            nn.init.normal_(self.raw_weight, mean=float(init_raw), std=0.05)

    def forward(self, x):
        return F.linear(x, positive(self.raw_weight), self.bias)


class LegacyMonotonicQualityNet(nn.Module):
    """Original unbounded Q branch retained only as an ablation/diagnostic."""
    def __init__(self, hidden=16, anchored=True, monotonic=True):
        super().__init__()
        self.anchored = anchored
        self.monotonic = monotonic
        if monotonic:
            self.net = nn.Sequential(
                PositiveLinear(1, hidden, init_raw=None), nn.Softplus(),
                PositiveLinear(hidden, hidden, init_raw=None), nn.Softplus(),
                PositiveLinear(hidden, 1, init_raw=None),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(1, hidden), nn.Softplus(),
                nn.Linear(hidden, hidden), nn.Softplus(),
                nn.Linear(hidden, 1),
            )

    def raw(self, q):
        return self.net(q.reshape(-1, 1)).squeeze(-1)

    def forward(self, q):
        y = self.raw(q)
        if self.anchored:
            one = torch.ones(1, device=q.device, dtype=q.dtype)
            y = y - self.raw(one).reshape(1)
        return y


# Backwards-compatible alias for older imports/checkpoints.
MonotonicQualityNet = LegacyMonotonicQualityNet


class BoundedMonotonicQualityNet(nn.Module):
    """Stable monotonic Q-efficiency branch with a parametric skip path.

    Let x=1-Q be the quality deficit. The monotonic model uses

        r(x) = a_+ x + h_+(x)-h_+(0),
        Delta_Q(Q) = -soft_cap(r(x)),

    where a_+ >= 0 and h_+ is a positive-weight MLP.  The direct skip path is
    initialized from the fitted parametric effective-data model, so epoch 0 is
    already close to the strong SL2 baseline.  The neural residual then learns
    only nonlinear improvements.  soft_cap is exactly the identity below
    ``soft_threshold`` and smoothly saturates at ``s_max`` above it, preventing
    the old exponential blow-up without flattening the empirically relevant
    range.
    """
    def __init__(self, hidden=8, s_max=6.0, soft_threshold=3.0,
                 anchored=True, monotonic=True, init_slope=1.0):
        super().__init__()
        self.hidden = int(hidden)
        self.s_max = float(s_max)
        self.soft_threshold = float(min(soft_threshold, s_max - 1e-6))
        self.anchored = bool(anchored)
        self.monotonic = bool(monotonic)
        self.init_slope = float(max(init_slope, 1e-6))
        # softplus(raw_skip) = init_slope.  This gives a strong, non-vanishing
        # gradient path and makes the initial network reproduce SL2 closely.
        raw_skip = math.log(math.expm1(self.init_slope)) if self.init_slope < 50 else self.init_slope
        self.raw_skip = nn.Parameter(torch.tensor(raw_skip, dtype=torch.float32))
        if monotonic:
            # Start the residual branch very close to zero; the parametric skip
            # carries the initial Q effect and the MLP learns only deviations.
            self.net = nn.Sequential(
                PositiveLinear(1, hidden, init_raw=-5.0), nn.Softplus(),
                PositiveLinear(hidden, hidden, init_raw=-5.0), nn.Softplus(),
                PositiveLinear(hidden, 1, init_raw=-5.0),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(1, hidden), nn.Softplus(),
                nn.Linear(hidden, hidden), nn.Softplus(),
                nn.Linear(hidden, 1),
            )
            # Near-zero residual so the unconstrained ablation starts from the
            # same parametric Q baseline rather than from a random large shift.
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def scale(self):
        # Compatibility helper used by diagnostics/metadata.
        return torch.tensor(self.s_max, dtype=self.raw_skip.dtype, device=self.raw_skip.device)

    def skip_slope(self):
        return F.softplus(self.raw_skip)

    def raw_residual(self, x):
        return self.net(x.reshape(-1, 1)).squeeze(-1)

    def _soft_cap_positive(self, r):
        """Identity for r<=threshold, smooth saturation to s_max afterwards."""
        t = self.soft_threshold
        span = max(self.s_max - t, 1e-6)
        tail = t + span * torch.tanh((r - t) / span)
        return torch.where(r <= t, r, tail)

    def forward(self, q):
        q = torch.clamp(q, 0.0, 1.0)
        x = 1.0 - q
        if self.monotonic:
            r = self.skip_slope() * x
            rr = self.raw_residual(x)
            if self.anchored:
                zero = torch.zeros(1, device=q.device, dtype=q.dtype)
                rr = rr - self.raw_residual(zero).reshape(1)
            r = torch.clamp(r + rr, min=0.0)
            return -self._soft_cap_positive(r)

        # Unconstrained ablation: retain the same parametric skip, allow an
        # unrestricted residual, but still cap the final correction for safety.
        residual = self.raw_residual(x)
        if self.anchored:
            zero = torch.zeros(1, device=q.device, dtype=q.dtype)
            residual = residual - self.raw_residual(zero).reshape(1)
        raw_delta = -self.skip_slope() * x + residual
        return torch.clamp(raw_delta, min=-self.s_max, max=0.0)


class ScaleConditionedMonotonicQualityNet(nn.Module):
    """Scale-conditioned monotonic quality-efficiency correction.

    The branch models Delta_Q(Q, D), not Delta_Q(Q) alone.  Let x=1-Q and
    z=(log D - mean)/std.  The quality deficit is

        r(x,z) = s(z) x + sum_k c_k(z) phi_k(x),
        Delta_Q(Q,D) = -soft_cap(r),

    where s(z)>0, c_k(z)>=0 and each phi_k is monotone increasing with
    phi_k(0)=0.  Therefore Delta_Q(1,D)=0 and d Delta_Q / dQ >= 0 for every
    D, while the magnitude of the quality effect is allowed to vary with data
    scale.  The direct slope contains an explicit Q x log(D) interaction and a
    small neural residual.
    """
    requires_d = True

    def __init__(self, hidden=8, d_hidden=8, n_basis=6, s_max=8.0,
                 soft_threshold=4.0, init_slope=1.0, init_d_coef=0.0,
                 logd_mean=0.0, logd_std=1.0):
        super().__init__()
        self.hidden=int(hidden); self.d_hidden=int(d_hidden); self.n_basis=int(n_basis)
        self.s_max=float(s_max); self.soft_threshold=float(min(soft_threshold,s_max-1e-6))
        self.init_slope=float(max(init_slope,1e-6)); self.init_d_coef=float(init_d_coef)
        self.register_buffer('logd_mean',torch.tensor(float(logd_mean),dtype=torch.float32))
        self.register_buffer('logd_std',torch.tensor(float(max(logd_std,1e-6)),dtype=torch.float32))

        raw_base=math.log(math.expm1(self.init_slope)) if self.init_slope < 50 else self.init_slope
        self.raw_base_slope=nn.Parameter(torch.tensor(raw_base,dtype=torch.float32))
        self.d_linear=nn.Parameter(torch.tensor(float(self.init_d_coef),dtype=torch.float32))
        self.d_encoder=nn.Sequential(
            nn.Linear(1,d_hidden),nn.Tanh(),
            nn.Linear(d_hidden,d_hidden),nn.Tanh(),
        )
        self.slope_residual=nn.Linear(d_hidden,1)
        nn.init.zeros_(self.slope_residual.weight); nn.init.zeros_(self.slope_residual.bias)

        # Monotone Q basis.  Positive basis slopes guarantee monotonicity in x=1-Q.
        basis_init=math.log(math.expm1(0.5))
        self.raw_basis_w=nn.Parameter(torch.full((n_basis,),basis_init,dtype=torch.float32))
        self.basis_bias=nn.Parameter(torch.linspace(-1.0,1.0,n_basis))
        self.coeff_head=nn.Linear(d_hidden,n_basis)
        nn.init.zeros_(self.coeff_head.weight)
        nn.init.constant_(self.coeff_head.bias,-6.0)  # near-zero nonlinear residual at epoch 0

    def _z(self,d):
        d=torch.clamp(d.reshape(-1),min=1e-12)
        return ((torch.log(d)-self.logd_mean.to(d))/self.logd_std.to(d)).reshape(-1,1)

    def _soft_cap_positive(self,r):
        t=self.soft_threshold; span=max(self.s_max-t,1e-6)
        tail=t+span*torch.tanh((r-t)/span)
        return torch.where(r<=t,r,tail)

    def slope(self,d):
        z=self._z(d); h=self.d_encoder(z)
        raw=self.raw_base_slope + self.d_linear*z.squeeze(-1) + self.slope_residual(h).squeeze(-1)
        return F.softplus(raw)

    def forward(self,q,d):
        q=torch.clamp(q.reshape(-1),0.0,1.0); x=1.0-q
        z=self._z(d); h=self.d_encoder(z)
        raw_slope=self.raw_base_slope + self.d_linear*z.squeeze(-1) + self.slope_residual(h).squeeze(-1)
        slope=F.softplus(raw_slope)
        w=F.softplus(self.raw_basis_w).reshape(1,-1)
        b=self.basis_bias.reshape(1,-1)
        xx=x.reshape(-1,1)
        phi=F.softplus(w*xx+b)-F.softplus(b)
        coeff=F.softplus(self.coeff_head(h))
        residual=torch.mean(coeff*phi,dim=1)
        r=torch.clamp(slope*x+residual,min=0.0)
        return -self._soft_cap_positive(r)


class ScaleConditionedUnconstrainedQualityNet(nn.Module):
    """Two-input Q,D neural ablation without monotonicity constraints.

    It is anchored at Q=1 for each D, but the derivative with respect to Q is
    unrestricted.  The final correction is clipped only for numerical safety.
    """
    requires_d=True
    def __init__(self, hidden=16, s_max=8.0, logd_mean=0.0, logd_std=1.0):
        super().__init__(); self.hidden=int(hidden); self.s_max=float(s_max)
        self.register_buffer('logd_mean',torch.tensor(float(logd_mean),dtype=torch.float32))
        self.register_buffer('logd_std',torch.tensor(float(max(logd_std,1e-6)),dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(2,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        for m in self.net:
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0.0,std=0.03); nn.init.zeros_(m.bias)
    def _z(self,d):
        d=torch.clamp(d.reshape(-1),min=1e-12)
        return (torch.log(d)-self.logd_mean.to(d))/self.logd_std.to(d)
    def raw(self,q,d):
        z=self._z(d); x=torch.stack([q.reshape(-1),z],dim=1); return self.net(x).squeeze(-1)
    def forward(self,q,d):
        q=torch.clamp(q.reshape(-1),0.0,1.0); d=d.reshape(-1)
        y=self.raw(q,d)-self.raw(torch.ones_like(q),d)
        return torch.clamp(y,min=-self.s_max,max=0.0)


def quality_delta(branch,q,d):
    """Call either the new scale-conditioned branch or legacy Q-only branches."""
    if getattr(branch,'requires_d',False):
        return branch(q,d)
    return branch(q)

def build_quality_correction_observations(df, params, eps=1e-12):
    """Derive observed Delta_Q from an already-fitted classic scaling backbone."""
    out = df.copy()
    E, A, B, alpha, beta = [float(params[k]) for k in ["E", "A", "B", "alpha", "beta"]]
    n = out["N_params_B"].to_numpy(float)
    d = out["D_tokens_B"].to_numpy(float)
    loss = out["val_loss"].to_numpy(float)
    q = out["Q_score"].to_numpy(float)
    scale_term = E + A * np.power(np.maximum(n, eps), -alpha)
    observed_data_term = loss - scale_term
    classic_data_term = B * np.power(np.maximum(d, eps), -beta)
    valid = (
        np.isfinite(q) & np.isfinite(observed_data_term) & np.isfinite(classic_data_term)
        & (observed_data_term > eps) & (classic_data_term > eps) & (beta > 0)
    )
    delta = np.full(len(out), np.nan, dtype=float)
    delta[valid] = -(1.0 / beta) * np.log(observed_data_term[valid] / classic_data_term[valid])
    out["scaling_nondata_term"] = scale_term
    out["observed_data_term"] = observed_data_term
    out["classic_data_term"] = classic_data_term
    out["delta_Q_obs"] = delta
    out["q_correction_valid"] = valid
    out["q_correction_physical"] = valid & np.isfinite(delta) & (delta <= 1e-8)
    return out


class MLPCompositionNet(nn.Module):
    def __init__(self,n_parts=17,hidden=(32,16),use_clr=True,anchored=True,p_ref=None):
        super().__init__(); self.n_parts=n_parts; self.use_clr=use_clr; self.anchored=anchored
        dims=[n_parts,*hidden,1]; layers=[]
        for a,b in zip(dims[:-2],dims[1:-1]): layers += [nn.Linear(a,b),nn.GELU()]
        layers += [nn.Linear(dims[-2],dims[-1])]
        self.net=nn.Sequential(*layers)
        pref=np.ones(n_parts)/n_parts if p_ref is None else np.asarray(p_ref,float)
        self.register_buffer("p_ref",torch.tensor(pref,dtype=torch.float32))
    @staticmethod
    def _clr(p,eps=1e-6):
        p=(torch.clamp(p,min=0)+eps); p=p/p.sum(dim=-1,keepdim=True); lp=torch.log(p); return lp-lp.mean(dim=-1,keepdim=True)
    def encode(self,p): return self._clr(p) if self.use_clr else p
    def raw(self,p): return self.net(self.encode(p)).squeeze(-1)
    def forward(self,p):
        y=self.raw(p)
        if self.anchored:
            ref=self.p_ref.to(p).reshape(1,-1).expand(max(1,p.shape[0]),-1)
            # anchor is a scalar function value; first element suffices
            anchor=self.raw(ref[:1]).reshape(1)
            y=y-anchor
        return y


class AttentionBlock(nn.Module):
    def __init__(self,d_model=32,n_heads=4,d_ff=64,dropout=0.1):
        super().__init__()
        self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_ff,d_model))
        self.drop=nn.Dropout(dropout)
    def forward(self,x,return_attention=False):
        a,w=self.attn(x,x,x,need_weights=True,average_attn_weights=False)
        x=self.norm1(x+self.drop(a)); x=self.norm2(x+self.drop(self.ff(x)))
        return (x,w) if return_attention else x


class AttentionCompositionNet(nn.Module):
    def __init__(self,n_parts=17,d_model=32,n_heads=4,n_layers=2,d_ff=64,dropout=0.1,
                 use_clr=True,use_embedding=True,anchored=True,p_ref=None):
        super().__init__(); self.n_parts=n_parts; self.use_clr=use_clr; self.use_embedding=use_embedding; self.anchored=anchored
        self.value_proj=nn.Linear(1,d_model)
        self.domain_embedding=nn.Parameter(torch.randn(n_parts,d_model)*0.02) if use_embedding else None
        self.blocks=nn.ModuleList([AttentionBlock(d_model,n_heads,d_ff,dropout) for _ in range(n_layers)])
        self.pool_query=nn.Parameter(torch.randn(d_model)*0.02)
        self.head=nn.Sequential(nn.Linear(d_model,16),nn.GELU(),nn.Linear(16,1))
        pref=np.ones(n_parts)/n_parts if p_ref is None else np.asarray(p_ref,float)
        self.register_buffer("p_ref",torch.tensor(pref,dtype=torch.float32))
        self.d_model=d_model
    @staticmethod
    def _clr(p,eps=1e-6):
        p=torch.clamp(p,min=0)+eps; p=p/p.sum(dim=-1,keepdim=True); lp=torch.log(p); return lp-lp.mean(dim=-1,keepdim=True)
    def tokens(self,p):
        v=self._clr(p) if self.use_clr else p
        x=self.value_proj(v.unsqueeze(-1))
        if self.domain_embedding is not None: x=x+self.domain_embedding.unsqueeze(0)
        return x
    def raw(self,p,return_attention=False):
        x=self.tokens(p); att=[]
        for block in self.blocks:
            if return_attention:
                x,w=block(x,True); att.append(w)
            else: x=block(x,False)
        score=(x*self.pool_query.reshape(1,1,-1)).sum(-1)/math.sqrt(self.d_model)
        a=torch.softmax(score,dim=1); pooled=(x*a.unsqueeze(-1)).sum(1)
        y=self.head(pooled).squeeze(-1)
        if return_attention: return y,att,a
        return y
    def forward(self,p):
        y=self.raw(p)
        if self.anchored:
            anchor=self.raw(self.p_ref.to(p).reshape(1,-1)).reshape(1)
            y=y-anchor
        return y
    def attention_maps(self,p):
        y,att,pool=self.raw(p,True); return att,pool


class MixtureLossWrapper(nn.Module):
    def __init__(self,branch:nn.Module,beta:float):
        super().__init__(); self.branch=branch; self.beta=float(beta)
        self.a0=nn.Parameter(torch.tensor(0.0)); self.c0=nn.Parameter(torch.tensor(1.0))
    def forward(self,p):
        delta=self.branch(p); a=F.softplus(self.a0); c=F.softplus(self.c0)
        return a+c*torch.exp(-self.beta*delta)


def scaling_with_quality_torch(n,d,q,params,branch:nn.Module):
    E,A,B,alpha,beta=[float(params[k]) for k in ["E","A","B","alpha","beta"]]
    delta=quality_delta(branch,q,d)
    return E + A*torch.pow(torch.clamp(n,min=1e-12),-alpha) + B*torch.pow(torch.clamp(d,min=1e-12)*torch.exp(delta),-beta)


def train_quality_loss_net(branch,params,train,valid=None,epochs=1500,lr=1e-3,weight_decay=1e-5,patience=120,seed=42,device="cpu",show_progress=True):
    torch.manual_seed(seed); branch=branch.to(device)
    opt=torch.optim.Adam(branch.parameters(),lr=lr,weight_decay=weight_decay)
    def tensors(df):
        return tuple(torch.tensor(df[c].to_numpy(float),dtype=torch.float32,device=device) for c in ["N_params_B","D_tokens_B","Q_score","val_loss"])
    tr=tensors(train); va=tensors(valid) if valid is not None and len(valid) else None
    best=None; bad=0; hist=[]
    progress = trange(1, epochs+1, desc="Q2 quality net", unit="epoch", leave=False, disable=not show_progress)
    for ep in progress:
        branch.train(); opt.zero_grad(); pred=scaling_with_quality_torch(tr[0],tr[1],tr[2],params,branch); loss=F.huber_loss(pred,tr[3],delta=0.1); loss.backward(); opt.step()
        with torch.no_grad():
            branch.eval(); vl=float(F.huber_loss(scaling_with_quality_torch(va[0],va[1],va[2],params,branch),va[3],delta=0.1)) if va else float(loss)
        hist.append((ep,float(loss.detach()),vl))
        if ep == 1 or ep % 25 == 0: progress.set_postfix(train=f"{float(loss.detach()):.4g}", valid=f"{vl:.4g}")
        if best is None or vl<best[0]-1e-8:
            best=(vl,{k:v.detach().cpu().clone() for k,v in branch.state_dict().items()},ep); bad=0
        else: bad+=1
        if bad>=patience: break
    branch.load_state_dict(best[1]); return branch,hist,best[2]


def _quality_loss_rmse(branch, raw_df, params, device="cpu"):
    if raw_df is None or len(raw_df) == 0:
        return float("nan")
    with torch.no_grad():
        n=torch.tensor(raw_df["N_params_B"].to_numpy(float),dtype=torch.float32,device=device)
        d=torch.tensor(raw_df["D_tokens_B"].to_numpy(float),dtype=torch.float32,device=device)
        q=torch.tensor(raw_df["Q_score"].to_numpy(float),dtype=torch.float32,device=device)
        y=torch.tensor(raw_df["val_loss"].to_numpy(float),dtype=torch.float32,device=device)
        pred=scaling_with_quality_torch(n,d,q,params,branch)
        return float(torch.sqrt(torch.mean((pred-y)**2)).cpu())


def train_quality_net(branch, train_obs, valid_obs=None, epochs=1500, lr=5e-4,
                      weight_decay=1e-6, patience=150, seed=42, device="cpu",
                      show_progress=True, grad_clip=1.0, physical_only=True,
                      select_params=None, select_valid_raw=None):
    """Stage A: residual supervision for the bounded Q branch.

    Training minimizes Huber loss on Delta_Q observations.  For the main model,
    impossible positive Delta_Q observations are excluded from this stage because
    the structural prior enforces Delta_Q<=0.  If a validation Loss table and
    scaling parameters are supplied, checkpoint selection is based on final-loss
    RMSE, with epoch 0 included.  This prevents residual pretraining from making
    the strong parametric initialization worse on B7.
    """
    torch.manual_seed(seed)
    branch = branch.to(device)

    def tensors(df):
        mask=df["q_correction_valid"].astype(bool)
        if physical_only and "q_correction_physical" in df:
            mask &= df["q_correction_physical"].astype(bool)
        d=df.loc[mask].dropna(subset=["Q_score","delta_Q_obs"])
        q=torch.tensor(d["Q_score"].to_numpy(float),dtype=torch.float32,device=device)
        ds=torch.tensor(d["D_tokens_B"].to_numpy(float),dtype=torch.float32,device=device)
        y=torch.tensor(d["delta_Q_obs"].to_numpy(float),dtype=torch.float32,device=device)
        return q,ds,y

    tr_q,tr_d,tr_y=tensors(train_obs)
    if tr_q.numel()==0:
        raise ValueError("No valid residual-supervision observations for the Q branch.")
    va=tensors(valid_obs) if valid_obs is not None and len(valid_obs) else None
    if va is not None and va[0].numel()==0: va=None

    opt=torch.optim.Adam(branch.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=40,min_lr=1e-6)

    def selection_metric():
        if select_params is not None and select_valid_raw is not None and len(select_valid_raw):
            return _quality_loss_rmse(branch,select_valid_raw,select_params,device)
        with torch.no_grad():
            return float(F.huber_loss(quality_delta(branch,va[0],va[1]),va[2],delta=0.25)) if va is not None else float(F.huber_loss(quality_delta(branch,tr_q,tr_d),tr_y,delta=0.25))

    # Epoch 0 is a valid candidate.  With init_slope from SL2 this is already a
    # strong baseline and gives a strict no-regression fallback on B7.
    best_metric=selection_metric()
    best=(best_metric,{k:v.detach().cpu().clone() for k,v in branch.state_dict().items()},0)
    bad=0; hist=[]
    progress=trange(1,epochs+1,desc="Q2 Q residual pretrain",unit="epoch",leave=False,disable=not show_progress)
    for ep in progress:
        branch.train(); opt.zero_grad(); pred=quality_delta(branch,tr_q,tr_d)
        loss=F.huber_loss(pred,tr_y,delta=0.25)
        loss.backward(); torch.nn.utils.clip_grad_norm_(branch.parameters(),max_norm=float(grad_clip)); opt.step()
        branch.eval()
        with torch.no_grad():
            vl=float(F.huber_loss(quality_delta(branch,va[0],va[1]),va[2],delta=0.25)) if va is not None else float(loss.detach())
        scheduler.step(vl); lr_now=float(opt.param_groups[0]["lr"]); sel=selection_metric()
        hist.append(("residual_pretrain",ep,float(loss.detach()),vl,sel,lr_now))
        if ep==1 or ep%25==0:
            slope_text=""
            if hasattr(branch,"skip_slope"):
                slope_text=f"{float(branch.skip_slope().detach().cpu()):.3g}"
            elif hasattr(branch,"slope"):
                slope_text=f"{float(branch.slope(torch.median(tr_d).reshape(1)).detach().cpu()[0]):.3g}"
            progress.set_postfix(train=f"{float(loss.detach()):.4g}",valid=f"{vl:.4g}",select=f"{sel:.4g}",slope=slope_text)
        if sel < best[0]-1e-7:
            best=(sel,{k:v.detach().cpu().clone() for k,v in branch.state_dict().items()},ep); bad=0
        else: bad+=1
        if bad>=patience: break
    branch.load_state_dict(best[1])
    return branch,hist,best[2]


def finetune_quality_loss_net(branch, params, train_raw, valid_raw=None, epochs=800,
                              lr=1e-4, weight_decay=1e-6, patience=120, seed=42,
                              device="cpu", show_progress=True, grad_clip=1.0):
    """Stage B: fine-tune the already bounded branch on the actual Loss scale.

    The scaling-law backbone stays frozen.  Validation RMSE is the checkpoint
    criterion and epoch 0 is included, so fine-tuning cannot make B7 worse than
    the incoming residual-pretrained model.
    """
    torch.manual_seed(seed); branch=branch.to(device)
    def tensors(df):
        return tuple(torch.tensor(df[c].to_numpy(float),dtype=torch.float32,device=device) for c in ["N_params_B","D_tokens_B","Q_score","val_loss"])
    tr=tensors(train_raw); va=tensors(valid_raw) if valid_raw is not None and len(valid_raw) else None
    opt=torch.optim.Adam(branch.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=35,min_lr=1e-6)
    def rmse_now(data):
        with torch.no_grad():
            pred=scaling_with_quality_torch(data[0],data[1],data[2],params,branch)
            return float(torch.sqrt(torch.mean((pred-data[3])**2)).cpu())
    initial=rmse_now(va if va is not None else tr)
    best=(initial,{k:v.detach().cpu().clone() for k,v in branch.state_dict().items()},0); bad=0; hist=[]
    progress=trange(1,epochs+1,desc="Q2 Q loss fine-tune",unit="epoch",leave=False,disable=not show_progress)
    for ep in progress:
        branch.train(); opt.zero_grad(); pred=scaling_with_quality_torch(tr[0],tr[1],tr[2],params,branch)
        loss=F.huber_loss(pred,tr[3],delta=0.05); loss.backward(); torch.nn.utils.clip_grad_norm_(branch.parameters(),max_norm=float(grad_clip)); opt.step()
        branch.eval(); sel=rmse_now(va if va is not None else tr); scheduler.step(sel); lr_now=float(opt.param_groups[0]["lr"])
        hist.append(("loss_finetune",ep,float(loss.detach()),sel,sel,lr_now))
        if ep==1 or ep%25==0:
            slope_text=""
            if hasattr(branch,"skip_slope"):
                slope_text=f"{float(branch.skip_slope().detach().cpu()):.3g}"
            elif hasattr(branch,"slope"):
                slope_text=f"{float(branch.slope(tr[1].median().reshape(1)).detach().cpu()[0]):.3g}"
            progress.set_postfix(train=f"{float(loss.detach()):.4g}",rmse=f"{sel:.4g}",slope=slope_text)
        if sel < best[0]-1e-7:
            best=(sel,{k:v.detach().cpu().clone() for k,v in branch.state_dict().items()},ep); bad=0
        else: bad+=1
        if bad>=patience: break
    branch.load_state_dict(best[1])
    return branch,hist,best[2]

def train_mixture_net(wrapper,train_p,train_y,valid_p=None,valid_y=None,epochs=2000,lr=1e-3,weight_decay=1e-5,patience=150,seed=42,device="cpu",show_progress=True):
    torch.manual_seed(seed); wrapper=wrapper.to(device)
    P=torch.tensor(np.asarray(train_p,float),dtype=torch.float32,device=device); Y=torch.tensor(np.asarray(train_y,float),dtype=torch.float32,device=device)
    VP=torch.tensor(np.asarray(valid_p,float),dtype=torch.float32,device=device) if valid_p is not None else None
    VY=torch.tensor(np.asarray(valid_y,float),dtype=torch.float32,device=device) if valid_y is not None else None
    opt=torch.optim.Adam(wrapper.parameters(),lr=lr,weight_decay=weight_decay); best=None; bad=0; hist=[]
    progress = trange(1, epochs+1, desc="Q2 mixture net", unit="epoch", leave=False, disable=not show_progress)
    for ep in progress:
        wrapper.train(); opt.zero_grad(); pred=wrapper(P); loss=F.huber_loss(pred,Y,delta=0.1); loss.backward(); opt.step()
        wrapper.eval()
        with torch.no_grad(): vl=float(F.huber_loss(wrapper(VP),VY,delta=0.1)) if VP is not None else float(loss)
        hist.append((ep,float(loss.detach()),vl))
        if ep == 1 or ep % 25 == 0: progress.set_postfix(train=f"{float(loss.detach()):.4g}", valid=f"{vl:.4g}")
        if best is None or vl<best[0]-1e-8:
            best=(vl,{k:v.detach().cpu().clone() for k,v in wrapper.state_dict().items()},ep); bad=0
        else: bad+=1
        if bad>=patience: break
    wrapper.load_state_dict(best[1]); return wrapper,hist,best[2]
