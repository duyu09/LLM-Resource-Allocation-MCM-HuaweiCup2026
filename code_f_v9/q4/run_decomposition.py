from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.io_utils import ensure_dir,read_csv,write_csv,dump_joblib
from common.plotting import setup_style,savefig,auto_break_barplot
from q4.decomposition import prepare_panel,time_split_metrics,AdditiveSplineGAM,yearly_shapley


def main():
    ap=argparse.ArgumentParser(description="Q4 scale-vs-technology decomposition: linear baseline, GAM, Shapley")
    ap.add_argument("--processed",type=Path,required=True); ap.add_argument("--capability",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("results/q4_decomposition"))
    a=ap.parse_args(); setup_style(); out=ensure_dir(a.out); figs=ensure_dir(out/"figures"); models=ensure_dir(out/"models")
    cap=read_csv(a.capability/"q4_capability_scores.csv"); lb=read_csv(a.processed/"C"/"q4_leaderboard.csv"); epoch=read_csv(a.processed/"C"/"q4_epoch_models.csv"); lb["submission_date"]=pd.to_datetime(lb.submission_date,errors="coerce"); epoch["publication_date"]=pd.to_datetime(epoch.publication_date,errors="coerce")
    panel=prepare_panel(cap,lb,epoch); write_csv(panel,out/"q4_model_panel.csv")
    features=["logN","logD","logC","time_months"]; complete=panel.dropna(subset=[*features,"cap_pca","model_type_bucket","submission_date"]).copy()
    if len(complete)<20: raise ValueError(f"Only {len(complete)} exact/high-confidence matched models have N,D,C,t and capability; inspect q4_model_panel.csv / matching rules before decomposition.")
    rows=[]
    for kind in ["linear","gam"]:
        met,_=time_split_metrics(complete,features,kind); rows.append({"model":kind,**met})
    simp=complete.dropna(subset=["logC","logD_N_ratio","time_months","model_type_bucket"]).copy()
    met_simple,_=time_split_metrics(simp,["logC","logD_N_ratio","time_months"],"gam")
    rows.append({"model":"gam_simplified_collinearity_check",**met_simple})
    write_csv(pd.DataFrame(rows),out/"q4_scale_tech_model_comparison.csv")
    gam=AdditiveSplineGAM(features,n_knots=6).fit(complete,complete.cap_pca); dump_joblib(gam,models/"scale_tech_gam.joblib")
    gam_simple=AdditiveSplineGAM(["logC","logD_N_ratio","time_months"],n_knots=6).fit(simp,simp.cap_pca); dump_joblib(gam_simple,models/"scale_tech_gam_simplified.joblib")

    # Quantify whether the time/technology trend survives the N,D,C multicollinearity reparameterization.
    tlo=max(float(complete.time_months.quantile(.02)),float(simp.time_months.quantile(.02))); thi=min(float(complete.time_months.quantile(.98)),float(simp.time_months.quantile(.98)))
    tgrid=np.linspace(tlo,thi,120)
    typ=str(complete.model_type_bucket.mode().iloc[0])
    mbase={c:float(complete[c].median()) for c in features}; mbase["model_type_bucket"]=typ
    main_grid=pd.DataFrame([mbase]*len(tgrid)); main_grid["time_months"]=tgrid; main_pred=gam.predict(main_grid)
    sbase={c:float(simp[c].median()) for c in ["logC","logD_N_ratio","time_months"]}; sbase["model_type_bucket"]=typ
    simple_grid=pd.DataFrame([sbase]*len(tgrid)); simple_grid["time_months"]=tgrid; simple_pred=gam_simple.predict(simple_grid)
    pear=float(np.corrcoef(main_pred,simple_pred)[0,1]) if np.std(main_pred)>0 and np.std(simple_pred)>0 else np.nan
    spear=float(pd.Series(main_pred).corr(pd.Series(simple_pred),method="spearman"))
    robust=pd.DataFrame([{"pearson_time_trend":pear,"spearman_time_trend":spear,
                          "main_delta_end_minus_start":float(main_pred[-1]-main_pred[0]),
                          "simplified_delta_end_minus_start":float(simple_pred[-1]-simple_pred[0]),
                          "same_direction":bool(np.sign(main_pred[-1]-main_pred[0])==np.sign(simple_pred[-1]-simple_pred[0])),
                          "n_main":len(complete),"n_simplified":len(simp)}])
    write_csv(robust,out/"q4_tech_trend_robustness.csv")
    write_csv(pd.DataFrame({"time_months":tgrid,"main_gam_tech_trend":main_pred,"simplified_gam_tech_trend":simple_pred}),out/"q4_tech_trend_curves.csv")

    shap=yearly_shapley(gam,complete); write_csv(shap,out/"q4_yearly_shapley_decomposition.csv")
    # Partial dependence by varying one feature around medians/mode.
    base={c:float(complete[c].median()) for c in features}; base["model_type_bucket"]=str(complete.model_type_bucket.mode().iloc[0]); pdrows=[]
    for c in features:
        grid=np.linspace(complete[c].quantile(.02),complete[c].quantile(.98),100); tmp=pd.DataFrame([base]*len(grid)); tmp[c]=grid; pred=gam.predict(tmp)
        for xv,yv in zip(grid,pred):pdrows.append({"feature":c,"x":xv,"partial_prediction":yv})
    pdd=pd.DataFrame(pdrows); write_csv(pdd,out/"q4_gam_partial_dependence.csv")

    # figures
    cmp=pd.DataFrame(rows); fig,axes,_=auto_break_barplot(data=cmp,x="model",y="rmse",color="#4C72B0",figsize=(7,5),title="Time-block validation: linear vs GAM"); savefig(figs/"q4d_01_linear_vs_gam.png",fig)
    if len(shap):
        sl=shap.melt(id_vars=["year_from","year_to"],value_vars=["phi_scale","phi_tech"],var_name="component",value_name="contribution"); sl["period"]=sl.year_from.astype(str)+"→"+sl.year_to.astype(str); fig,axes,_=auto_break_barplot(data=sl,x="period",y="contribution",hue="component",figsize=(10,5),title="Shapley decomposition of capability growth",rotation=30,zero_line=True); savefig(figs/"q4d_02_shapley_contributions.png",fig)
    g=sns.FacetGrid(pdd,col="feature",col_wrap=2,sharex=False,sharey=True,height=3.2); g.map_dataframe(sns.lineplot,x="x",y="partial_prediction"); g.set_titles("{col_name}"); g.figure.suptitle("GAM partial dependence",y=1.02); savefig(figs/"q4d_03_gam_partial_dependence.png",g.figure)
    tc=pd.DataFrame({"time_months":tgrid,"main_GAM":main_pred,"simplified_GAM":simple_pred}).melt("time_months",var_name="model",value_name="technology_trend")
    fig,ax=plt.subplots(figsize=(8,5)); sns.lineplot(data=tc,x="time_months",y="technology_trend",hue="model",ax=ax); ax.set_title("Technology-trend robustness to scale reparameterization"); savefig(figs/"q4d_04_tech_trend_robustness.png",fig)
    match=panel.epoch_match_method.value_counts(dropna=False).rename_axis("match_method").reset_index(name="count"); write_csv(match,out/"q4_epoch_match_summary.csv")
    print(f"Q4 decomposition complete: {out}; complete modeling rows={len(complete)}")
if __name__=="__main__":main()
