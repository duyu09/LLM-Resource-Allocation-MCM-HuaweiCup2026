from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from common.io_utils import ensure_dir, read_csv, write_csv, write_json, read_json
from common.compositional import aitchison_center
from common.random_utils import set_seed
from q2.scaling_baseline import fit_scaling_law, choose_groups
from q2.neural_models import ScaleConditionedMonotonicQualityNet, AttentionCompositionNet, MixtureLossWrapper, train_quality_net, finetune_quality_loss_net, train_mixture_net, build_quality_correction_observations


def cluster_resample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    groups=choose_groups(df).astype(str)
    keys=groups.unique(); sampled=rng.choice(keys,size=len(keys),replace=True)
    chunks=[]
    for b,k in enumerate(sampled):
        x=df.loc[groups.eq(k)].copy(); x["_boot_cluster"]=b; chunks.append(x)
    return pd.concat(chunks,ignore_index=True)


def row_resample(df: pd.DataFrame,rng):
    idx=rng.integers(0,len(df),size=len(df)); return df.iloc[idx].reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser(description="Q2 cluster/bootstrap uncertainty for Q3 robust optimization.")
    ap.add_argument("--processed",type=Path,required=True)
    ap.add_argument("--q2",type=Path,required=True,help="Completed Q2 result directory")
    ap.add_argument("--out",type=Path,default=None)
    ap.add_argument("--n-boot",type=int,default=1000)
    ap.add_argument("--refit-neural",action="store_true",help="Also refit Q and p branches for each bootstrap replicate (computationally heavy).")
    ap.add_argument("--epochs-q",type=int,default=300)
    ap.add_argument("--epochs-p",type=int,default=400)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args(); set_seed(args.seed)
    out=ensure_dir(args.out or (args.q2/"bootstrap")); ens=ensure_dir(out/"ensemble")
    sc=read_csv(args.processed/"B"/"q2_scaling.csv"); b1=sc[sc.source_id.eq("B1")].dropna(subset=["N_params_B","D_tokens_B","val_loss"])
    b6=sc[sc.source_id.eq("B6")].dropna(subset=["N_params_B","D_tokens_B","Q_score","val_loss"])
    b7=sc[sc.source_id.eq("B7")].dropna(subset=["N_params_B","D_tokens_B","Q_score","val_loss"])
    mix=read_csv(args.processed/"A"/"q1_mixture_loss.csv"); pcols=[c for c in mix if c.startswith("p_")]; lcols=[c for c in mix if c.startswith("loss_") and c != "loss_nature"]
    mt=mix[mix.pair_id.eq("A4_A5")].dropna(subset=pcols+lcols); P=mt[pcols].to_numpy(float); y=mt[lcols].mean(axis=1).to_numpy(float)
    rng=np.random.default_rng(args.seed); rows=[]
    for b in trange(args.n_boot, desc="Q2 cluster bootstrap", unit="rep"):
        brng=np.random.default_rng(rng.integers(0,2**32-1))
        b1b=cluster_resample(b1,brng); params,meta=fit_scaling_law(b1b,n_starts=12,seed=args.seed+b,show_progress=False)
        row={"bootstrap":b,**params.to_dict(),"fit_sse":meta["objective_sse"]}
        if args.refit_neural:
            rep=ensure_dir(ens/f"boot_{b:04d}"); write_json(params.to_dict(),rep/"scaling_params.json")
            qb=row_resample(b6,brng); qv=row_resample(b7,brng) if len(b7) else None
            qb_obs=build_quality_correction_observations(qb,params.to_dict())
            qv_obs=build_quality_correction_observations(qv,params.to_dict()) if qv is not None else None
            phys=qb_obs[qb_obs["q_correction_physical"].astype(bool)].dropna(subset=["Q_score","delta_Q_obs"])
            xx=phys["Q_score"].to_numpy(float)-1.0; yy=phys["delta_Q_obs"].to_numpy(float)
            init_slope=float(np.clip(np.sum(xx*yy)/(np.sum(xx*xx)+1e-12),1e-3,20.0)) if len(phys) else 1.0
            logd=np.log(np.maximum(qb["D_tokens_B"].to_numpy(float),1e-12)); logd_mean=float(logd.mean()); logd_std=float(max(logd.std(ddof=0),1e-6))
            qnet=ScaleConditionedMonotonicQualityNet(hidden=8,d_hidden=8,n_basis=6,s_max=8.0,soft_threshold=4.0,
                                                     init_slope=init_slope,init_d_coef=0.0,logd_mean=logd_mean,logd_std=logd_std)
            qnet,_,_=train_quality_net(qnet,qb_obs,qv_obs,args.epochs_q,lr=5e-4,patience=150,seed=args.seed+b,device=args.device,show_progress=False,physical_only=True,select_params=params.to_dict(),select_valid_raw=qv)
            qnet,_,_=finetune_quality_loss_net(qnet,params.to_dict(),qb,qv,epochs=max(200,args.epochs_q//2),lr=1e-4,patience=100,seed=args.seed+b+1,device=args.device,show_progress=False)
            torch.save({"state_dict":qnet.cpu().state_dict(),"architecture":"scale_conditioned_monotonic_qd_v3",
                        "hidden":8,"d_hidden":8,"n_basis":6,"s_max":8.0,"soft_threshold":4.0,
                        "init_slope":init_slope,"init_d_coef":0.0,"logd_mean":logd_mean,"logd_std":logd_std},rep/"q_branch.pt")
            idx=brng.integers(0,len(P),size=len(P)); Pb,yb=P[idx],y[idx]; pref=aitchison_center(Pb)
            # holdout from bootstrap sample for early stopping
            order=np.arange(len(Pb)); brng.shuffle(order); cut=max(1,int(.8*len(order))); it,iv=order[:cut],order[cut:]
            pn=AttentionCompositionNet(17,32,4,2,64,.1,True,True,True,pref); wrap=MixtureLossWrapper(pn,params.beta)
            wrap,_,_=train_mixture_net(wrap,Pb[it],yb[it],Pb[iv],yb[iv],args.epochs_p,seed=args.seed+b,device=args.device,show_progress=False)
            torch.save({"state_dict":pn.cpu().state_dict(),"p_ref":pref.tolist(),"n_parts":17,"d_model":32,"n_heads":4,"n_layers":2,"d_ff":64,"dropout":.1,"use_clr":True,"use_embedding":True,"anchored":True},rep/"p_branch.pt")
        rows.append(row)
        if (b+1)%25==0 or b==0:
            write_csv(pd.DataFrame(rows),out/"q2_bootstrap_scaling_params.csv")
    write_csv(pd.DataFrame(rows),out/"q2_bootstrap_scaling_params.csv")
    write_json({"n_boot":args.n_boot,"refit_neural":args.refit_neural,"seed":args.seed},out/"bootstrap_meta.json")
    print(f"Q2 bootstrap complete: {out}")

if __name__=="__main__": main()
