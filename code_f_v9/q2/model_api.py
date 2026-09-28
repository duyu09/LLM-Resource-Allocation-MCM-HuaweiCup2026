from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch

from common.io_utils import read_json
from q2.neural_models import LegacyMonotonicQualityNet, BoundedMonotonicQualityNet, ScaleConditionedMonotonicQualityNet, AttentionCompositionNet, quality_delta


class CEADNSLPredictor:
    def __init__(self,params,q_branch,p_branch,device="cpu"):
        self.params={k:float(params[k]) for k in ["E","A","B","alpha","beta"]}
        self.q_branch=q_branch.to(device).eval(); self.p_branch=p_branch.to(device).eval(); self.device=device
    def predict(self,N,D,Q,p):
        N=np.asarray(N,float); D=np.asarray(D,float); Q=np.asarray(Q,float); P=np.asarray(p,float)
        scalar=P.ndim==1 and N.ndim==0
        if P.ndim==1: P=P.reshape(1,-1)
        n=np.broadcast_to(N/1e9,(P.shape[0],)) if N.ndim==0 else N/1e9
        d=np.broadcast_to(D/1e9,(P.shape[0],)) if D.ndim==0 else D/1e9
        q=np.broadcast_to(Q,(P.shape[0],)) if Q.ndim==0 else Q
        with torch.no_grad():
            tq=torch.tensor(q,dtype=torch.float32,device=self.device); td=torch.tensor(d,dtype=torch.float32,device=self.device); tp=torch.tensor(P,dtype=torch.float32,device=self.device)
            dq=quality_delta(self.q_branch,tq,td).cpu().numpy(); dp=self.p_branch(tp).cpu().numpy()
        s=dq+dp; E,A,B,a,b=[self.params[k] for k in ["E","A","B","alpha","beta"]]
        pred=E+A*np.power(np.maximum(n,1e-12),-a)+B*np.power(np.maximum(d,1e-12)*np.exp(s),-b)
        return float(pred[0]) if scalar else pred
    def corrections(self,Q,p,D=None):
        P=np.asarray(p,float); scalar=P.ndim==1
        if P.ndim==1:P=P.reshape(1,-1)
        q=np.broadcast_to(np.asarray(Q,float),(P.shape[0],))
        if D is None:
            if getattr(self.q_branch,"requires_d",False):
                raise ValueError("D is required for the scale-conditioned quality branch.")
            d=np.ones(P.shape[0],float)
        else:
            darr=np.asarray(D,float); d=np.broadcast_to(darr/1e9,(P.shape[0],)) if darr.ndim==0 else darr/1e9
        with torch.no_grad():
            tq=torch.tensor(q,dtype=torch.float32,device=self.device); td=torch.tensor(d,dtype=torch.float32,device=self.device)
            dq=quality_delta(self.q_branch,tq,td).cpu().numpy()
            dp=self.p_branch(torch.tensor(P,dtype=torch.float32,device=self.device)).cpu().numpy()
        return (float(dq[0]),float(dp[0])) if scalar else (dq,dp)


def load_predictor(artifact_dir: str|Path,device="cpu",params_override=None):
    art=Path(artifact_dir)
    params=params_override or read_json(art/"scaling_params.json")
    qck=torch.load(art/"q_branch.pt",map_location=device,weights_only=False)
    arch=qck.get("architecture","legacy_unbounded")
    if arch == "scale_conditioned_monotonic_qd_v3":
        q=ScaleConditionedMonotonicQualityNet(
            hidden=qck.get("hidden",8),d_hidden=qck.get("d_hidden",8),n_basis=qck.get("n_basis",6),
            s_max=qck.get("s_max",8.0),soft_threshold=qck.get("soft_threshold",4.0),
            init_slope=qck.get("init_slope",1.0),init_d_coef=qck.get("init_d_coef",0.0),
            logd_mean=qck.get("logd_mean",0.0),logd_std=qck.get("logd_std",1.0),
        )
    elif arch in {"bounded_monotonic_residual","bounded_monotonic_residual_v2"}:
        q=BoundedMonotonicQualityNet(
            hidden=qck.get("hidden",8),
            s_max=qck.get("s_max",6.0 if arch.endswith("v2") else 3.0),
            soft_threshold=qck.get("soft_threshold",3.0),
            anchored=qck.get("anchored",True),
            monotonic=qck.get("monotonic",True),
            init_slope=qck.get("init_slope",1.0),
        )
    else:
        q=LegacyMonotonicQualityNet(hidden=qck.get("hidden",16),anchored=qck.get("anchored",True),monotonic=qck.get("monotonic",True))
    q.load_state_dict(qck["state_dict"])
    pck=torch.load(art/"p_branch.pt",map_location=device,weights_only=False)
    p=AttentionCompositionNet(n_parts=pck["n_parts"],d_model=pck["d_model"],n_heads=pck["n_heads"],n_layers=pck["n_layers"],d_ff=pck["d_ff"],dropout=pck["dropout"],use_clr=pck["use_clr"],use_embedding=pck["use_embedding"],anchored=pck["anchored"],p_ref=pck["p_ref"])
    p.load_state_dict(pck["state_dict"])
    return CEADNSLPredictor(params,q,p,device)
