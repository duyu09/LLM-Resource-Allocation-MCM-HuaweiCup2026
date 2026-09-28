from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from common.io_utils import ensure_dir,read_csv,write_csv,dump_joblib,load_joblib,write_json
from common.plotting import setup_style,savefig,auto_break_barplot
from common.metrics import weighted_regression_metrics
from q4.capability import BENCH_COLS
from q4.bridge import select_ispline_hyperparams,loocv_models,fit_all_models,monotonic_violation


def main():
    ap=argparse.ArgumentParser(description="Q4 Loss-to-Benchmark bridge: Linear/Spline/Isotonic/I-Spline + LOOCV")
    ap.add_argument("--processed",type=Path,required=True); ap.add_argument("--capability",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("results/q4_bridge")); ap.add_argument("--q3-results",type=Path,default=None); ap.add_argument("--fast",action="store_true")
    a=ap.parse_args();setup_style();out=ensure_dir(a.out);figs=ensure_dir(out/"figures");modelsdir=ensure_dir(out/"models")
    df=read_csv(a.processed/"C"/"q4_loss_benchmark_bridge.csv",required=["val_loss","comparability_bucket",*BENCH_COLS]); df=df.dropna(subset=["val_loss"]); x=df.val_loss.to_numpy(float); bucket=df.comparability_bucket.astype(str).to_numpy()
    allmods={}; metric_rows=[]; curve_rows=[]; hparams={}
    for target in tqdm(BENCH_COLS, desc="Q4 Loss→Benchmark bridges", unit="benchmark"):
        use=df.dropna(subset=[target,"val_loss"]); xx=use.val_loss.to_numpy(float); yy=use[target].to_numpy(float); bb=use.comparability_bucket.astype(str).to_numpy()
        omega,lam=select_ispline_hyperparams(xx,yy,bb); hparams[target]={"omega":omega,"lambda":lam}
        if a.fast:
            # fast still runs all models but skips LOOCV in favor of in-sample metrics
            mods,w=fit_all_models(xx,yy,bb,omega,lam); preds={name:m.predict(xx) for name,m in mods.items()}
        else:
            preds,w=loocv_models(xx,yy,bb,omega,lam); mods,_=fit_all_models(xx,yy,bb,omega,lam)
        allmods[target]=mods
        for name,p in preds.items():
            met=weighted_regression_metrics(yy,p,w); viol=monotonic_violation(mods[name],float(xx.min()),float(xx.max()))
            metric_rows.append({"benchmark":target,"model":name,"omega":omega,"lambda":lam,"monotonic_violation_rate":viol,**met})
        grid=np.linspace(xx.min(),xx.max(),200)
        for name,m in mods.items():
            for xv,yv in zip(grid,m.predict(grid)):curve_rows.append({"benchmark":target,"model":name,"loss":xv,"prediction":yv})
    write_csv(pd.DataFrame(metric_rows),out/"q4_bridge_model_comparison.csv"); write_csv(pd.DataFrame(curve_rows),out/"q4_bridge_curves.csv"); write_json(hparams,out/"q4_bridge_hyperparameters.json"); dump_joblib(allmods,modelsdir/"bridge_models.joblib")

    # Project Q3 optimal losses through six main I-spline bridges and PCA capability transform.
    q3=a.q3_results
    if q3 and q3.exists():
        q3df=pd.read_csv(q3); bundle=load_joblib(a.capability/"models"/"capability_pca.joblib"); rows=[]
        for _,r in q3df.iterrows():
            loss=float(r.expected_loss); vals={b:float(allmods[b]["i_spline"].predict([loss])[0]) for b in BENCH_COLS}; tmp=pd.DataFrame([vals]); cap=float(bundle.transform(tmp)[0]); rows.append({**r.to_dict(),**{f"pred_{k}":v for k,v in vals.items()},"pred_capability_pca":cap})
        write_csv(pd.DataFrame(rows),out/"q4_q3_loss_to_benchmark_projection.csv")

    # Figures: comparison + one panel per benchmark.
    met=pd.DataFrame(metric_rows); avg=met.groupby("model",as_index=False).rmse_w.mean().sort_values("rmse_w"); fig,axes,_=auto_break_barplot(data=avg,x="rmse_w",y="model",color="#4C72B0",figsize=(8,5),title="Loss→Benchmark model comparison (mean weighted RMSE)",xlabel="Weighted RMSE");savefig(figs/"q4b_01_bridge_comparison.png",fig)
    curves=pd.DataFrame(curve_rows)
    for target in BENCH_COLS:
        use=df.dropna(subset=[target,"val_loss"]); fig,ax=plt.subplots(figsize=(7,5));sns.scatterplot(data=use,x="val_loss",y=target,hue="comparability_bucket",s=45,ax=ax);sns.lineplot(data=curves[curves.benchmark.eq(target)],x="loss",y="prediction",hue="model",ax=ax);ax.set_title(f"Loss to {target}: baselines and monotone I-spline");savefig(figs/f"q4b_{target}.png",fig)
    print(f"Q4 bridge complete: {out}")
if __name__=="__main__":main()
