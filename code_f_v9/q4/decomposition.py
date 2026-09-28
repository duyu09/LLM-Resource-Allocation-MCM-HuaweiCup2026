from __future__ import annotations

import re
from pathlib import PurePosixPath
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm.auto import tqdm

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


def _simple_key(s):
    s = "" if pd.isna(s) else str(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _basename_name(s):
    """Return repository basename for names like org/model while preserving plain names."""
    s = "" if pd.isna(s) else str(s).strip().replace("\\", "/")
    return s.rsplit("/", 1)[-1].strip()


def _base_variant_name(s):
    """Remove common post-training/version suffixes for conservative base-model matching."""
    s = _basename_name(s).lower()
    # Strip only common terminal variant markers; do not remove size/version tokens.
    patterns = [
        r"[-_ ](?:instruct|instruction|chat|it|assistant|sft|dpo|rlhf|aligned|base|hf)$",
        r"[-_ ](?:instruct|chat|it|assistant|sft|dpo|rlhf|aligned|base|hf)[-_ ].*$",
    ]
    old = None
    while s and s != old:
        old = s
        for pat in patterns:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
    return s.strip("-_ ")


def _param_ok(lb_params_b, ep_params_b, ratio_limit=1.25):
    try:
        lp = float(lb_params_b)
        ep = float(ep_params_b)
    except Exception:
        return True
    if not (np.isfinite(lp) and np.isfinite(ep) and lp > 0 and ep > 0):
        return True
    return max(lp, ep) / min(lp, ep) <= ratio_limit


def _date_gap_days(lb_date, ep_date):
    try:
        a = pd.Timestamp(lb_date)
        b = pd.Timestamp(ep_date)
        if pd.isna(a) or pd.isna(b):
            return np.inf
        return abs((a - b).days)
    except Exception:
        return np.inf


def collapse_epoch(epoch: pd.DataFrame):
    x = epoch.copy()
    complete_cols = [c for c in ["training_compute_flop", "training_dataset_size_total", "parameters"] if c in x]
    x["_complete"] = x[complete_cols].notna().sum(axis=1) if complete_cols else 0
    x["publication_date"] = pd.to_datetime(x.get("publication_date"), errors="coerce")
    if "model_key" not in x.columns:
        x["model_key"] = x.get("model", pd.Series("", index=x.index)).map(_simple_key)
    x = x.sort_values(["model_key", "_complete", "publication_date"], ascending=[True, False, False])
    x = x.drop_duplicates("model_key", keep="first")
    return x.drop(columns="_complete").reset_index(drop=True)


def _choose_candidate(scores_row, candidates, idxs, require_param=False, ratio_limit=1.25):
    if not idxs:
        return None
    lbp = scores_row.get("params_B", np.nan)
    valid = []
    for j in idxs:
        ep = candidates.iloc[j]
        epp = ep.get("parameters_B", np.nan)
        if require_param and not _param_ok(lbp, epp, ratio_limit):
            continue
        # Prefer parameter agreement, then closer dates, then more complete compute/data/parameter row.
        if np.isfinite(pd.to_numeric(lbp, errors="coerce")) and np.isfinite(pd.to_numeric(epp, errors="coerce")) and float(lbp) > 0 and float(epp) > 0:
            param_err = abs(np.log(float(lbp) / float(epp)))
        else:
            param_err = 1e6
        date_gap = _date_gap_days(scores_row.get("submission_date"), ep.get("publication_date"))
        completeness = sum(pd.notna(ep.get(c, np.nan)) for c in ["training_compute_flop", "training_dataset_size_total", "parameters"])
        valid.append((param_err, date_gap, -completeness, j))
    return min(valid)[-1] if valid else None


def merge_capability_epoch(scores: pd.DataFrame, epoch: pd.DataFrame, fuzzy_threshold=88, param_ratio_limit=1.25):
    """Five-layer conservative leaderboard -> Epoch matcher.

    Layers:
      1) exact model_key
      2) exact repository basename (case-insensitive)
      3) normalized basename exact
      4) base-variant exact after stripping chat/instruct/etc., with parameter guard
      5) high-confidence fuzzy basename, with parameter guard

    Every accepted row records match method, score and matched Epoch model.
    """
    e = collapse_epoch(epoch).copy()
    s = scores.copy().reset_index(drop=True)
    e["_epoch_idx"] = np.arange(len(e))
    e["_model_raw"] = e.get("model", pd.Series("", index=e.index)).astype(str)
    e["_basename"] = e["_model_raw"].map(_basename_name)
    e["_basename_lower"] = e["_basename"].str.lower()
    e["_basename_key"] = e["_basename"].map(_simple_key)
    e["_base_variant_key"] = e["_basename"].map(_base_variant_name).map(_simple_key)

    def index_map(col):
        d = {}
        for j, v in enumerate(e[col].astype(str)):
            if not v:
                continue
            d.setdefault(v, []).append(j)
        return d

    key_map = index_map("model_key")
    basename_map = index_map("_basename_lower")
    norm_map = index_map("_basename_key")
    variant_map = index_map("_base_variant_key")

    match_rows = []
    for i, row in tqdm(s.iterrows(), total=len(s), desc="Q4 Epoch hierarchical matching", unit="model", leave=False):
        raw = str(row.get("model", ""))
        model_key = str(row.get("model_key", ""))
        basename = _basename_name(raw)
        basename_lower = basename.lower()
        basename_key = _simple_key(basename)
        variant_key = _simple_key(_base_variant_name(basename))

        chosen = None
        method = "unmatched"
        score = np.nan

        # Layer 1: strict model_key exact.
        idxs = key_map.get(model_key, [])
        chosen = _choose_candidate(row, e, idxs, require_param=False, ratio_limit=param_ratio_limit)
        if chosen is not None:
            method, score = "L1_exact_model_key", 100.0

        # Layer 2: HuggingFace/repository basename exact.
        if chosen is None and basename_lower:
            idxs = basename_map.get(basename_lower, [])
            chosen = _choose_candidate(row, e, idxs, require_param=False, ratio_limit=param_ratio_limit)
            if chosen is not None:
                method, score = "L2_basename_exact", 100.0

        # Layer 3: normalized basename exact.
        if chosen is None and basename_key:
            idxs = norm_map.get(basename_key, [])
            chosen = _choose_candidate(row, e, idxs, require_param=False, ratio_limit=param_ratio_limit)
            if chosen is not None:
                method, score = "L3_normalized_basename_exact", 100.0

        # Layer 4: strip post-training suffixes, but require compatible parameter scale.
        if chosen is None and variant_key:
            idxs = variant_map.get(variant_key, [])
            chosen = _choose_candidate(row, e, idxs, require_param=True, ratio_limit=param_ratio_limit)
            if chosen is not None:
                method, score = "L4_base_variant_param_guard", 97.0

        # Layer 5: fuzzy basename with parameter guard. Use basename only, never org/repo prefix.
        if chosen is None and fuzz is not None and basename_key:
            best = []
            for j, ck in enumerate(e["_basename_key"].astype(str)):
                if not ck:
                    continue
                if not _param_ok(row.get("params_B", np.nan), e.iloc[j].get("parameters_B", np.nan), param_ratio_limit):
                    continue
                sc = float(fuzz.ratio(basename_key, ck))
                if sc >= fuzzy_threshold:
                    best.append((-sc, _date_gap_days(row.get("submission_date"), e.iloc[j].get("publication_date")), j))
            if best:
                neg_sc, _, chosen = min(best)
                method, score = "L5_fuzzy_basename_param_guard", -neg_sc

        match_rows.append({"_score_row": i, "_epoch_idx": chosen, "epoch_match_method": method, "epoch_match_score": score})

    matches = pd.DataFrame(match_rows).set_index("_score_row")
    out = s.copy()
    original_score_cols = set(out.columns)
    out["epoch_match_method"] = matches["epoch_match_method"].reindex(out.index).to_numpy()
    out["epoch_match_score"] = matches["epoch_match_score"].reindex(out.index).to_numpy()

    # Materialize the selected Epoch rows in one pass.  Destination names are
    # determined against the ORIGINAL leaderboard/capability columns, not against
    # columns added during the loop; this avoids the 旧版 bug where only the first
    # matched row received N/D/C and later rows were accidentally written to
    # *_epoch columns.
    selected = []
    for i in out.index:
        j = matches.at[i, "_epoch_idx"]
        selected.append(None if pd.isna(j) else e.iloc[int(j)])
    out["epoch_matched_model"] = [pd.NA if ep is None else ep.get("model", pd.NA) for ep in selected]
    out["epoch_matched_model_key"] = [pd.NA if ep is None else ep.get("model_key", pd.NA) for ep in selected]
    for c in epoch.columns:
        if c in {"model", "model_key"}:
            continue
        dest = c if c not in original_score_cols else f"{c}_epoch"
        out[dest] = [np.nan if ep is None else ep.get(c, np.nan) for ep in selected]
    return out


class AdditiveSplineGAM:
    def __init__(self,numeric_cols,categorical_col="model_type_bucket",n_knots=6,degree=3,alphas=None):
        self.numeric_cols=list(numeric_cols); self.categorical_col=categorical_col; self.n_knots=n_knots; self.degree=degree; self.alphas=np.logspace(-3,3,25) if alphas is None else alphas
        self.splines={}; self.onehot=None; self.scaler=None; self.model=None; self.slices={}
    def _fit_design(self,df):
        parts=[]; start=0
        for c in self.numeric_cols:
            vals=df[[c]].to_numpy(float); uniq=np.unique(vals[np.isfinite(vals)]); nk=max(2,min(self.n_knots,len(uniq)))
            sp=SplineTransformer(n_knots=nk,degree=self.degree,include_bias=False,extrapolation="linear").fit(vals); Z=sp.transform(vals); self.splines[c]=sp; self.slices[c]=slice(start,start+Z.shape[1]); start+=Z.shape[1]; parts.append(Z)
        try: oh=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        except TypeError: oh=OneHotEncoder(handle_unknown="ignore",sparse=False)
        C=oh.fit_transform(df[[self.categorical_col]].astype(str)); self.onehot=oh; self.slices[self.categorical_col]=slice(start,start+C.shape[1]); parts.append(C)
        X=np.c_[*parts]; self.scaler=StandardScaler().fit(X); return self.scaler.transform(X)
    def _design(self,df):
        parts=[self.splines[c].transform(df[[c]].to_numpy(float)) for c in self.numeric_cols]; parts.append(self.onehot.transform(df[[self.categorical_col]].astype(str))); return self.scaler.transform(np.c_[*parts])
    def fit(self,df,y):
        X=self._fit_design(df); self.model=RidgeCV(alphas=self.alphas).fit(X,np.asarray(y,float)); return self
    def predict(self,df): return self.model.predict(self._design(df))


def prepare_panel(capability,leaderboard,epoch):
    base=leaderboard.merge(capability[["model_key","cap_raw_mean","cap_z_mean","cap_pca","pc2"]],on="model_key",how="inner")
    panel=merge_capability_epoch(base,epoch)
    def pick(*names):
        for n in names:
            if n in panel.columns:return panel[n]
        return pd.Series(np.nan,index=panel.index)
    panel["N"] = pd.to_numeric(pick("parameters","parameters_epoch"),errors="coerce")
    panel["D"] = pd.to_numeric(pick("training_dataset_size_total","training_dataset_size_total_epoch"),errors="coerce")
    panel["C"] = pd.to_numeric(pick("training_compute_flop","training_compute_flop_epoch"),errors="coerce")
    panel["open_weights_bool_final"] = pick("open_weights_bool","open_weights_bool_epoch")
    panel["submission_date"]=pd.to_datetime(panel["submission_date"],errors="coerce")
    t0=panel.submission_date.min(); panel["time_months"]=(panel.submission_date-t0).dt.days/30.4375
    for c,src in [("logN","N"),("logD","D"),("logC","C")]: panel[c]=np.log10(pd.to_numeric(panel[src],errors="coerce").where(pd.to_numeric(panel[src],errors="coerce")>0))
    panel["logD_N_ratio"]=panel.logD-panel.logN
    return panel


def time_split_metrics(df,features,model_kind="gam"):
    x=df.dropna(subset=[*features,"cap_pca","submission_date","model_type_bucket"]).sort_values("submission_date"); cut=max(5,int(.7*len(x))); tr=x.iloc[:cut]; te=x.iloc[cut:]
    if len(te)<3:return {},None
    if model_kind=="linear":
        Xtr=pd.get_dummies(tr[features+['model_type_bucket']],columns=['model_type_bucket'],drop_first=False); Xte=pd.get_dummies(te[features+['model_type_bucket']],columns=['model_type_bucket'],drop_first=False).reindex(columns=Xtr.columns,fill_value=0)
        m=LinearRegression().fit(Xtr,tr.cap_pca); pred=m.predict(Xte)
    else:
        m=AdditiveSplineGAM(features).fit(tr,tr.cap_pca); pred=m.predict(te)
    met={"n_train":len(tr),"n_test":len(te),"rmse":float(np.sqrt(mean_squared_error(te.cap_pca,pred))),"mae":float(mean_absolute_error(te.cap_pca,pred)),"r2":float(r2_score(te.cap_pca,pred))}
    return met,m


def yearly_shapley(model,df,scale_cols=("logN","logD","logC"),time_col="time_months",type_col="model_type_bucket"):
    x=df.dropna(subset=[*scale_cols,time_col,type_col,"submission_date"]).copy(); x["year"]=x.submission_date.dt.year; states={}
    for y,g in x.groupby("year"):
        st={c:float(g[c].median()) for c in scale_cols}; st[time_col]=float(g[time_col].median()); st[type_col]=str(g[type_col].mode().iloc[0]); states[int(y)]=st
    years=sorted(states); rows=[]
    def pred(scale_state,tech_state):
        row={c:scale_state[c] for c in scale_cols}; row[time_col]=tech_state[time_col]; row[type_col]=tech_state[type_col]; return float(model.predict(pd.DataFrame([row]))[0])
    for y0,y1 in zip(years[:-1],years[1:]):
        S0,S1=states[y0],states[y1]; f00=pred(S0,S0); f10=pred(S1,S0); f01=pred(S0,S1); f11=pred(S1,S1); phi_s=.5*((f10-f00)+(f11-f01)); phi_t=.5*((f01-f00)+(f11-f10)); delta=f11-f00
        rows.append({"year_from":y0,"year_to":y1,"delta_capability":delta,"phi_scale":phi_s,"phi_tech":phi_t,"scale_share":phi_s/delta if abs(delta)>1e-12 else np.nan,"tech_share":phi_t/delta if abs(delta)>1e-12 else np.nan})
    return pd.DataFrame(rows)
