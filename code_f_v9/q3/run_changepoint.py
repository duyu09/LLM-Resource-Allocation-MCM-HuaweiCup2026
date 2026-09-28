from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from common.io_utils import ensure_dir,read_csv,write_csv
from common.plotting import setup_style,savefig


def bic(y,pred,k):
    y=np.asarray(y,float); pred=np.asarray(pred,float); n=len(y); rss=max(float(np.sum((y-pred)**2)),1e-30); return n*np.log(rss/n)+k*np.log(n)

def segmented_fit(x,y,c):
    X=np.c_[np.ones(len(x)),x,np.maximum(x-c,0)]; coef=np.linalg.lstsq(X,y,rcond=None)[0]; return X@coef,coef

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--q3",type=Path,required=True); ap.add_argument("--out",type=Path,default=None)
    a=ap.parse_args(); setup_style(); out=ensure_dir(a.out or a.q3); figs=ensure_dir(out/"figures")
    df=read_csv(a.q3/"q3_budget_scan.csv").sort_values("budget").reset_index(drop=True); df["logC"]=np.log10(df.budget); df["logN"]=np.log10(df.N_opt); df["logD"]=np.log10(df.D_opt)
    feats=["logN","logD","Q_opt","r_train","r_quality","r_attn","mixture_entropy"]; Y=StandardScaler().fit_transform(df[feats])
    try:
        import ruptures as rpt
        pen=3*np.log(len(df)); bkps=rpt.Pelt(model="rbf",min_size=3).fit(Y).predict(pen=pen); cps=[b for b in bkps if b<len(df)]
    except Exception as e:
        # Conservative fallback: candidate point with largest multivariate first-difference norm.
        d=np.linalg.norm(np.diff(Y,axis=0),axis=1); cps=[int(np.argmax(d)+1)] if len(d) else []
    cp_rows=[]
    for cp in cps: cp_rows.append({"index":cp,"budget":float(df.loc[cp,"budget"]),"log10_budget":float(df.loc[cp,"logC"])})
    write_csv(pd.DataFrame(cp_rows),out/"q3_pelt_changepoints.csv")
    # Segmented regression confirmation for logN/logD/Q.
    rows=[]
    x=df.logC.to_numpy(float)
    for target in ["logN","logD","Q_opt"]:
        y=df[target].to_numpy(float); lin=LinearRegression().fit(x.reshape(-1,1),y); p0=lin.predict(x.reshape(-1,1)); b0=bic(y,p0,2)
        for cp in cps:
            c=float(x[cp]); pp,coef=segmented_fit(x,y,c); bp=bic(y,pp,3); rows.append({"target":target,"change_index":cp,"change_log10_budget":c,"BIC_single":b0,"BIC_piecewise":bp,"BIC_improvement":b0-bp,"supported_gt10":(b0-bp)>10,"slope_before":coef[1],"slope_change":coef[2]})
    conf=pd.DataFrame(rows); write_csv(conf,out/"q3_segmented_regression_confirmation.csv")
    fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=df,x="logC",y="Q_opt",marker="o",ax=ax); [ax.axvline(df.loc[c,"logC"],ls="--") for c in cps]; ax.set_title("Structural transition candidates in optimal quality"); savefig(figs/"q3_11_changepoints_Q.png",fig)
    long=df.melt(id_vars="logC",value_vars=["r_train","r_quality","r_attn"],var_name="component",value_name="share"); fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=long,x="logC",y="share",hue="component",ax=ax); [ax.axvline(df.loc[c,"logC"],ls="--",lw=1) for c in cps]; ax.set_title("Cost-share structural transitions"); savefig(figs/"q3_12_changepoints_costshare.png",fig)
    print(f"Q3 changepoint analysis complete: {out}; candidates={len(cps)}")
if __name__=="__main__":main()
