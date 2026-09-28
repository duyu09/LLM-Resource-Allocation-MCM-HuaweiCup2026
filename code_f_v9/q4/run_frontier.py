from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm, trange
from common.io_utils import ensure_dir, read_csv, write_csv, dump_joblib
from common.plotting import setup_style, savefig, auto_break_barplot
from common.metrics import pinball_loss
from q4.frontier import (
    QuantileGAM, DeepQuantileRegressor, MonotonicDeepQuantileRegressor,
    prepare_frontier_panel, compute_frontier_series, fit_compute_growth,
    rolling_empirical_forecast, is_open_weights, license_bucket,
)
from q4.capability import BENCH_COLS


MODEL_LABELS = {
    "rolling_q95": "F-A Rolling Q95",
    "linear_quantile": "F-B Linear QR",
    "quantile_gam": "F-C Quantile GAM",
    "deep_quantile_mlp": "F-D Deep Quantile MLP",
    "monotonic_deep_quantile": "F-E Monotonic DQN",
    "monotonic_interaction_deep_quantile": "F-F Monotonic DQN + t×C",
}


def linear_qr_fit(df, tau=.95, use_compute=True):
    cols = ["time_months"] + (["logC"] if use_compute else [])
    X = df[cols].to_numpy(float)
    m = QuantileRegressor(quantile=tau, alpha=1e-4, solver="highs").fit(X, df.cap_pca)
    return m, cols


def actual_window_q95(df, date, months=3):
    half_days = max(1, int(round(months * 30.4375 / 2)))
    lo = date - pd.DateOffset(days=half_days); hi = date + pd.DateOffset(days=half_days)
    u = df[(df.submission_date >= lo) & (df.submission_date <= hi)]
    return float(u.cap_pca.quantile(.95)) if len(u) >= 3 else np.nan


def fit_deep_models(df, use_compute, seed=42, fast=False):
    epochs = 300 if fast else 1200
    patience = 45 if fast else 120
    dmlp = DeepQuantileRegressor(.95, max_epochs=epochs, patience=patience, seed=seed).fit(df, df.cap_pca, use_compute=use_compute)
    mdqn = MonotonicDeepQuantileRegressor(.95, interaction=False, max_epochs=epochs, patience=patience, seed=seed+1).fit(df, df.cap_pca, use_compute=use_compute)
    midqn = MonotonicDeepQuantileRegressor(.95, interaction=True, max_epochs=epochs, patience=patience, seed=seed+2).fit(df, df.cap_pca, use_compute=use_compute)
    return dmlp, mdqn, midqn


def make_prediction_row(tm, logc, typ, use_compute):
    d = {"time_months": [tm], "model_type_bucket": [typ]}
    if use_compute: d["logC"] = [logc]
    return pd.DataFrame(d)


def main():
    ap = argparse.ArgumentParser(description="Q4 frontier model comparison, rolling-origin backtest, slowdown scenarios")
    ap.add_argument("--decomposition", type=Path, required=True)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/q4_frontier"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    setup_style(); out = ensure_dir(a.out); figs = ensure_dir(out/"figures"); models = ensure_dir(out/"models")

    panel = read_csv(a.decomposition/"q4_model_panel.csv"); panel["submission_date"] = pd.to_datetime(panel.submission_date, errors="coerce")
    screening = panel.copy(); screening["license_bucket"] = screening.get("hub_license", pd.Series("", index=screening.index)).map(license_bucket); screening["open_weights_selected"] = screening.apply(is_open_weights, axis=1)
    keep = [c for c in ["model","model_key","submission_date","model_type_bucket","hub_license","license_bucket","open_weights_bool_final","open_weights_selected","epoch_match_method","epoch_match_score","epoch_matched_model"] if c in screening.columns]
    write_csv(screening[keep], out/"q4_open_model_screening.csv")

    front, t0 = prepare_frontier_panel(panel)
    train = front.dropna(subset=["time_months","cap_pca","model_type_bucket"]).copy(); use_compute = train.logC.notna().sum() >= 30
    fit = train.dropna(subset=["logC"]) if use_compute else train
    if len(fit) < 20: raise ValueError(f"Too few open-weight capability rows for frontier modeling: {len(fit)}")

    # F-B/F-C plus new deep F-D/F-E/F-F models fitted on the full eligible panel.
    qgam = QuantileGAM(.95).fit(fit, fit.cap_pca, use_compute=use_compute); dump_joblib(qgam, models/"F-C_quantile_gam.joblib")
    lqr, lcols = linear_qr_fit(fit, .95, use_compute); dump_joblib((lqr,lcols), models/"F-B_linear_quantile.joblib")
    with tqdm(total=3, desc="Q4 deep frontier full fits", unit="model") as pbar:
        dmlp = DeepQuantileRegressor(.95, max_epochs=(300 if a.fast else 1200), patience=(45 if a.fast else 120), seed=a.seed).fit(fit, fit.cap_pca, use_compute=use_compute); pbar.update(1)
        mdqn = MonotonicDeepQuantileRegressor(.95, interaction=False, max_epochs=(300 if a.fast else 1200), patience=(45 if a.fast else 120), seed=a.seed+1).fit(fit, fit.cap_pca, use_compute=use_compute); pbar.update(1)
        midqn = MonotonicDeepQuantileRegressor(.95, interaction=True, max_epochs=(300 if a.fast else 1200), patience=(45 if a.fast else 120), seed=a.seed+2).fit(fit, fit.cap_pca, use_compute=use_compute); pbar.update(1)
    dump_joblib(dmlp, models/"F-D_deep_quantile_mlp.joblib"); dump_joblib(mdqn, models/"F-E_monotonic_deep_quantile.joblib"); dump_joblib(midqn, models/"F-F_monotonic_interaction_deep_quantile.joblib")
    mono_diag = pd.concat([
        mdqn.monotonicity_check(fit).assign(model="F-E_monotonic_deep_quantile"),
        midqn.monotonicity_check(fit).assign(model="F-F_monotonic_interaction_deep_quantile"),
    ], ignore_index=True)
    write_csv(mono_diag, out/"q4_frontier_deep_monotonicity.csv")

    # Epoch compute frontier is needed both for historical backtesting and future scenarios.
    epoch = read_csv(a.processed/"C"/"q4_epoch_models.csv"); epoch["publication_date"] = pd.to_datetime(epoch.publication_date, errors="coerce")
    series = compute_frontier_series(epoch, .9); write_csv(series, out/"q4_compute_frontier_quarterly.csv")

    # Rolling-origin backtest at 6 and 12 months. All six F-A ... F-F methods use only past data at each cutoff.
    dates = pd.date_range(fit.submission_date.min()+pd.DateOffset(months=18), fit.submission_date.max()-pd.DateOffset(months=6), freq="QS")
    rows = []
    if a.fast: dates = dates[-min(4, len(dates)):]
    for cutoff_idx, cutoff in enumerate(tqdm(dates, desc="Q4 rolling-origin backtest (F-A..F-F)", unit="cutoff")):
        tr = fit[fit.submission_date <= cutoff].copy()
        if len(tr) < 20: continue
        qg = QuantileGAM(.95).fit(tr, tr.cap_pca, use_compute=use_compute); lm, _ = linear_qr_fit(tr, .95, use_compute)
        dm, mq, mi = fit_deep_models(tr, use_compute, seed=a.seed + 100*cutoff_idx, fast=a.fast)
        for h in [6, 12]:
            target = cutoff + pd.DateOffset(months=h); actual = actual_window_q95(fit, target, 3)
            if not np.isfinite(actual): continue
            tm = (target-t0).days/30.4375; typ = str(tr.model_type_bucket.mode().iloc[0])
            if use_compute:
                hist_series = series[series.quarter <= cutoff].copy()
                if len(hist_series) >= 2:
                    try:
                        g_hist,_,_,_,last_logc_hist = fit_compute_growth(hist_series,24); logc = float(last_logc_hist + g_hist*h)
                    except Exception: logc = float(tr.logC.quantile(.9))
                else: logc = float(tr.logC.quantile(.9))
                X = np.array([[tm, logc]])
            else:
                logc = np.nan; X = np.array([[tm]])
            new = make_prediction_row(tm, logc, typ, use_compute)
            preds = {
                "rolling_q95": rolling_empirical_forecast(tr, target),
                "linear_quantile": float(lm.predict(X)[0]),
                "quantile_gam": float(qg.predict(new)[0]),
                "deep_quantile_mlp": float(dm.predict(new)[0]),
                "monotonic_deep_quantile": float(mq.predict(new)[0]),
                "monotonic_interaction_deep_quantile": float(mi.predict(new)[0]),
            }
            for name,p in preds.items():
                rows.append({"cutoff":cutoff,"horizon_months":h,"target_date":target,"model":name,"model_label":MODEL_LABELS[name],"prediction":p,"actual_q95":actual,"abs_error":abs(p-actual),"sq_error":(p-actual)**2,"pinball":pinball_loss([actual],[p],.95),"frontier_covered":bool(actual<=p)})
    back = pd.DataFrame(rows); write_csv(back, out/"q4_frontier_backtest.csv")
    if len(back):
        cmp = back.groupby(["model","model_label","horizon_months"],as_index=False).agg(MAE=("abs_error","mean"),RMSE=("sq_error",lambda s:float(np.sqrt(np.mean(s)))),Pinball=("pinball","mean"),FrontierCoverage=("frontier_covered","mean"),N=("prediction","size"))
        write_csv(cmp, out/"q4_frontier_model_comparison.csv")
        rank = back.groupby(["model","model_label"],as_index=False).agg(MAE=("abs_error","mean"),RMSE=("sq_error",lambda s:float(np.sqrt(np.mean(s)))),Pinball=("pinball","mean"),FrontierCoverage=("frontier_covered","mean"),N=("prediction","size"))
        rank["pinball_rank"] = rank.Pinball.rank(method="min"); rank["mae_rank"] = rank.MAE.rank(method="min"); rank = rank.sort_values(["pinball_rank","mae_rank","model"]).reset_index(drop=True)
        write_csv(rank, out/"q4_frontier_model_ranking.csv")

    # Compute-frontier growth and slowdown scenarios. Keep every old method and add F-D/F-E/F-F.
    g, intercept, tstart, last_date, last_logC = fit_compute_growth(series, 24)
    future = []; typ = str(fit.model_type_bucket.mode().iloc[0]); last_data = fit.submission_date.max()
    recent_cut = last_data - pd.DateOffset(months=12); recent = fit[fit.submission_date >= recent_cut]
    naive_q95 = float((recent if len(recent) else fit).cap_pca.quantile(.95))
    for factor,label in [(1.0,"normal"),(.75,"mild_slowdown"),(.5,"main_slowdown"),(.25,"severe_slowdown")]:
        for h in [12,24]:
            date = last_data + pd.DateOffset(months=h); tm = (date-t0).days/30.4375; logc = last_logC + factor*g*h
            new = make_prediction_row(tm, logc, typ, use_compute)
            X = np.array([[tm,logc]]) if use_compute else np.array([[tm]])
            preds = {
                "rolling_q95": naive_q95,
                "linear_quantile": float(lqr.predict(X)[0]),
                "quantile_gam": float(qgam.predict(new)[0]),
                "deep_quantile_mlp": float(dmlp.predict(new)[0]),
                "monotonic_deep_quantile": float(mdqn.predict(new)[0]),
                "monotonic_interaction_deep_quantile": float(midqn.predict(new)[0]),
            }
            for name,pred in preds.items():
                future.append({"scenario":label,"growth_factor":factor,"horizon_months":h,"forecast_date":date,"model":name,"model_label":MODEL_LABELS[name],"log10_compute_frontier":logc,"capability_frontier_q95":pred})
    future = pd.DataFrame(future); write_csv(future, out/"q4_frontier_future_scenarios.csv")

    # Quarter-block bootstrap retained from the previous design as the Quantile-GAM reference uncertainty band.
    # Deep models participate in the formal rolling-origin comparison and future-scenario table;
    # refitting three neural models 1000 times would be computationally excessive and is not
    # silently substituted for the original uncertainty analysis.
    nboot = min(a.bootstrap,30) if a.fast else a.bootstrap
    rng = np.random.default_rng(a.seed); fit2 = fit.copy(); fit2["quarter"] = fit2.submission_date.dt.to_period("Q").dt.start_time; quarters = fit2.quarter.unique(); boots=[]
    recent_series = series.sort_values("quarter").copy(); recent_series = recent_series[recent_series.quarter >= recent_series.quarter.max()-pd.DateOffset(months=24)].copy()
    if len(recent_series)<2: recent_series=series.tail(min(8,len(series))).copy()
    tc=(recent_series.quarter-recent_series.quarter.min()).dt.days.to_numpy(float)/30.4375; yc=recent_series.logC_q.to_numpy(float)
    creg=LinearRegression().fit(tc.reshape(-1,1),yc); cfit=creg.predict(tc.reshape(-1,1)); cres=yc-cfit; tlast=float(tc[-1])
    for b in trange(nboot, desc="Q4 frontier block bootstrap (F-C reference)", unit="rep"):
        qs=rng.choice(quarters,size=len(quarters),replace=True); parts=[fit2[fit2.quarter.eq(q)].copy() for q in qs]
        bs=pd.concat(parts,ignore_index=True) if parts else fit2.iloc[0:0].copy()
        if len(bs)<20: continue
        target_col="cap_pca"
        if all(c in bs.columns for c in BENCH_COLS):
            complete=bs.dropna(subset=BENCH_COLS).copy()
            if len(complete)>=20:
                Xb=complete[BENCH_COLS].to_numpy(float); sc=StandardScaler().fit(Xb); Zb=sc.transform(Xb); pc=PCA().fit_transform(Zb)[:,0]; raw=Xb.mean(axis=1); sg=1.0 if np.corrcoef(pc,raw)[0,1]>=0 else -1.0
                complete["cap_boot"]=pc*sg; bs=complete; target_col="cap_boot"
        try: bg=QuantileGAM(.95).fit(bs,bs[target_col],use_compute=use_compute)
        except Exception: continue
        if len(cres):
            yboot=cfit+rng.choice(cres,size=len(cres),replace=True); breg=LinearRegression().fit(tc.reshape(-1,1),yboot); gb=float(breg.coef_[0]); last_logc_b=float(breg.predict(np.array([[tlast]]))[0])
        else: gb=float(g); last_logc_b=float(last_logC)
        typ_b=str(bs.model_type_bucket.mode().iloc[0]) if bs.model_type_bucket.notna().any() else typ
        for h in [12,24]:
            date=last_data+pd.DateOffset(months=h); tm=(date-t0).days/30.4375; logc=last_logc_b+.5*gb*h; new=make_prediction_row(tm,logc,typ_b,use_compute)
            boots.append({"bootstrap":b,"horizon_months":h,"model":"quantile_gam","prediction":float(bg.predict(new)[0]),"compute_growth_per_month":gb,"future_log10_compute":logc,"pca_refit":target_col=="cap_boot"})
    boot=pd.DataFrame(boots); write_csv(boot,out/"q4_frontier_bootstrap.csv")
    if len(boot):
        ci=boot.groupby("horizon_months").prediction.quantile([.025,.5,.975]).unstack().reset_index(); ci.columns=["horizon_months","ci_low","median","ci_high"]; write_csv(ci,out/"q4_frontier_bootstrap_ci.csv")

    # Figures
    if len(back):
        cmp=read_csv(out/"q4_frontier_model_comparison.csv"); fig,axes,_=auto_break_barplot(data=cmp,x="horizon_months",y="MAE",hue="model_label",figsize=(12,6),title="Rolling-origin frontier forecast MAE: F-A to F-F"); savefig(figs/"q4f_01_backtest_mae.png",fig)
    # Facet by model so six methods x four scenarios remain readable.
    gfig=sns.relplot(data=future,x="horizon_months",y="capability_frontier_q95",hue="scenario",col="model_label",col_wrap=2,kind="line",marker="o",height=3.4,facet_kws={"sharey":True}); gfig.set_titles("{col_name}"); gfig.figure.suptitle("12/24-month open-model capability frontier scenarios: F-A to F-F",y=1.02); savefig(figs/"q4f_02_future_scenarios.png",gfig.figure)
    if len(boot):
        ci=read_csv(out/"q4_frontier_bootstrap_ci.csv"); fig,ax=plt.subplots(figsize=(8,5)); ax.fill_between(ci.horizon_months,ci.ci_low,ci.ci_high,alpha=.25); sns.lineplot(data=ci,x="horizon_months",y="median",marker="o",ax=ax); ax.set_title("Main slowdown: F-C Quantile GAM bootstrap reference uncertainty"); savefig(figs/"q4f_03_bootstrap_fan.png",fig)
    fig,ax=plt.subplots(figsize=(10,5)); sns.lineplot(data=series,x="quarter",y="logC_q",marker="o",ax=ax); ax.set_title("Quarterly 90th-percentile open-model training-compute frontier"); savefig(figs/"q4f_04_compute_frontier.png",fig)
    if len(back):
        rank=read_csv(out/"q4_frontier_model_ranking.csv"); fig,axes,_=auto_break_barplot(data=rank,x="model_label",y="Pinball",color="#4C72B0",figsize=(12,5),title="Overall rolling-origin pinball loss: F-A to F-F",rotation=25); savefig(figs/"q4f_05_model_ranking_pinball.png",fig)
    print(f"Q4 frontier complete: {out}; use_compute={use_compute}; fitted open models={len(fit)}")


if __name__ == "__main__":
    main()
