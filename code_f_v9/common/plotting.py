from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from pandas.api.types import is_numeric_dtype


def setup_style():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "font.family": "sans-serif",
    })


def savefig(path: str | Path, fig=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _normalize_bar_input(data, x, y, hue=None):
    """Return a dataframe + column names even when x/y are array-like."""
    if data is not None:
        return data.copy(), x, y, hue
    if isinstance(x, str) or isinstance(y, str):
        raise ValueError("When data=None, x and y must both be array-like, not column names.")
    df = pd.DataFrame({"__x": np.asarray(x), "__y": np.asarray(y)})
    hk = None
    if hue is not None:
        if isinstance(hue, str):
            raise ValueError("When data=None, hue must be array-like, not a column name.")
        df["__hue"] = np.asarray(hue)
        hk = "__hue"
    return df, "__x", "__y", hk


def _infer_bar_orientation(df, x, y, orient=None):
    if orient in {"h", "horizontal"}:
        return "h"
    if orient in {"v", "vertical"}:
        return "v"
    xn = is_numeric_dtype(df[x])
    yn = is_numeric_dtype(df[y])
    if xn and not yn:
        return "h"
    return "v"


def _summary_bar_values(df, x, y, hue, orient):
    num = x if orient == "h" else y
    cat = y if orient == "h" else x
    z = df.copy()
    z[num] = pd.to_numeric(z[num], errors="coerce")
    groups = [cat] + ([hue] if hue is not None else [])
    try:
        vals = z.groupby(groups, dropna=False, observed=False)[num].mean().to_numpy(float)
    except Exception:
        vals = z[num].to_numpy(float)
    return vals[np.isfinite(vals)]


def _tail_gap_candidate(magnitudes, ratio_threshold=8.0):
    a=np.sort(np.unique(np.asarray(magnitudes,float)))
    a=a[np.isfinite(a) & (a>0)]
    if len(a)<2: return None
    ratios=a[1:]/np.maximum(a[:-1],1e-15)
    i=int(np.argmax(ratios)); gap=float(ratios[i])
    if gap<ratio_threshold:
        return None
    return float(a[i]),float(a[i+1]),float(a[-1]),gap


def _auto_break_limits(values, ratio_threshold=8.0):
    """Detect an exploding positive or negative tail and return two axis segments.

    The main segment always retains zero (and therefore the ordinary small bars),
    while the tail segment shows the exploding bar(s) at their true numeric scale.
    """
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    if len(v)<3: return None
    pos=_tail_gap_candidate(v[v>0],ratio_threshold)
    neg=_tail_gap_candidate(np.abs(v[v<0]),ratio_threshold)
    candidates=[]
    if pos is not None:
        low,high,maxmag,gap=pos
        main_min=min(0.0,float(v.min())*1.06 if v.min()<0 else 0.0)
        candidates.append({"side":"high","gap_ratio":gap,
                           "main_min":main_min,"main_max":low*1.18,
                           "tail_min":high*0.92,"tail_max":maxmag*1.06})
    if neg is not None:
        low,high,maxmag,gap=neg
        main_max=max(0.0,float(v.max())*1.06 if v.max()>0 else 0.0)
        candidates.append({"side":"low","gap_ratio":gap,
                           "tail_min":-maxmag*1.06,"tail_max":-high*0.92,
                           "main_min":-low*1.18,"main_max":main_max})
    if not candidates: return None
    br=max(candidates,key=lambda z:z["gap_ratio"])
    if br["side"]=="high":
        if not (br["main_max"]>br["main_min"] and br["tail_min"]>br["main_max"] and br["tail_max"]>br["tail_min"]): return None
    else:
        if not (br["tail_max"]>br["tail_min"] and br["main_min"]>br["tail_max"] and br["main_max"]>br["main_min"]): return None
    return br


def _remove_legend(ax):
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def _draw_horizontal_break_marks(ax_left, ax_right, d=0.012):
    kw = dict(color="k", clip_on=False, linewidth=1.0)
    ax_left.plot((1-d, 1+d), (-d, +d), transform=ax_left.transAxes, **kw)
    ax_left.plot((1-d, 1+d), (1-d, 1+d), transform=ax_left.transAxes, **kw)
    ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, **kw)
    ax_right.plot((-d, +d), (1-d, 1+d), transform=ax_right.transAxes, **kw)


def _draw_vertical_break_marks(ax_top, ax_bottom, d=0.012):
    kw = dict(color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1-d, 1+d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_bottom.plot((-d, +d), (1-d, 1+d), transform=ax_bottom.transAxes, **kw)
    ax_bottom.plot((1-d, 1+d), (1-d, 1+d), transform=ax_bottom.transAxes, **kw)


def auto_break_barplot(*, data=None, x=None, y=None, hue=None, orient=None,
                       color=None, palette=None, order=None, hue_order=None,
                       figsize=(8, 5), title=None, xlabel=None, ylabel=None,
                       rotation=None, zero_line=False, xlim=None, ylim=None,
                       break_ratio=8.0, errorbar=None, **kwargs):
    """Seaborn barplot with an automatic broken numeric axis for explosive outliers.

    Every project bar chart should call this helper. If the aggregated bar heights
    contain a large positive multiplicative gap (default >=8x), the helper draws
    the same Seaborn barplot on two linked axes and inserts standard diagonal break
    marks. This keeps both the exploding item and the small bars visible while
    preserving the original numeric values. Otherwise a normal Seaborn barplot is
    returned.

    Returns: (fig, axes_tuple, break_info). break_info is None when no break is used.
    """
    df, xk, yk, hk = _normalize_bar_input(data, x, y, hue)
    ori = _infer_bar_orientation(df, xk, yk, orient)
    vals = _summary_bar_values(df, xk, yk, hk, ori)
    br = _auto_break_limits(vals, break_ratio)

    bar_kw = dict(data=df, x=xk, y=yk, hue=hk, orient=ori,
                  color=color, palette=palette, order=order, hue_order=hue_order,
                  errorbar=errorbar)
    bar_kw.update(kwargs)
    # Seaborn warns when palette=None is explicitly passed in some versions.
    bar_kw = {k: v for k, v in bar_kw.items() if v is not None}

    # Explicit caller axis limits take priority over auto-breaking.
    if xlim is not None or ylim is not None:
        br = None

    if br is None:
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(ax=ax, **bar_kw)
        if xlim is not None: ax.set_xlim(*xlim)
        if ylim is not None: ax.set_ylim(*ylim)
        if zero_line:
            (ax.axvline if ori == "h" else ax.axhline)(0, lw=1, color="0.25")
        if title: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if rotation is not None: ax.tick_params(axis="x", rotation=rotation)
        return fig, (ax,), None

    side=br["side"]
    if ori == "h":
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ratios=(3.2,1.25) if side=="high" else (1.25,3.2)
        gs=fig.add_gridspec(1,2,width_ratios=ratios,wspace=0.06)
        ax_left=fig.add_subplot(gs[0,0]); ax_right=fig.add_subplot(gs[0,1],sharey=ax_left)
        sns.barplot(ax=ax_left,**bar_kw); sns.barplot(ax=ax_right,**bar_kw)
        if side=="high":
            ax_left.set_xlim(br["main_min"],br["main_max"]); ax_right.set_xlim(br["tail_min"],br["tail_max"]); main_ax=ax_left; tail_ax=ax_right
        else:
            ax_left.set_xlim(br["tail_min"],br["tail_max"]); ax_right.set_xlim(br["main_min"],br["main_max"]); main_ax=ax_right; tail_ax=ax_left
        ax_left.spines["right"].set_visible(False); ax_right.spines["left"].set_visible(False)
        ax_right.tick_params(axis="y",left=False,labelleft=False); ax_right.set_ylabel("")
        ax_left.set_xlabel(""); ax_right.set_xlabel("")
        _remove_legend(tail_ax); _draw_horizontal_break_marks(ax_left,ax_right)
        if zero_line and main_ax.get_xlim()[0]<=0<=main_ax.get_xlim()[1]: main_ax.axvline(0,lw=1,color="0.25")
        if title: fig.suptitle(title)
        if xlabel is not None: fig.supxlabel(xlabel)
        if ylabel is not None: fig.supylabel(ylabel)
        if rotation is not None:
            ax_left.tick_params(axis="x",rotation=rotation); ax_right.tick_params(axis="x",rotation=rotation)
        out=dict(br); out["orientation"]="h"; return fig,(ax_left,ax_right),out

    fig=plt.figure(figsize=figsize,constrained_layout=True)
    ratios=(1.25,3.2) if side=="high" else (3.2,1.25)
    gs=fig.add_gridspec(2,1,height_ratios=ratios,hspace=0.06)
    ax_top=fig.add_subplot(gs[0,0]); ax_bottom=fig.add_subplot(gs[1,0],sharex=ax_top)
    sns.barplot(ax=ax_top,**bar_kw); sns.barplot(ax=ax_bottom,**bar_kw)
    if side=="high":
        ax_top.set_ylim(br["tail_min"],br["tail_max"]); ax_bottom.set_ylim(br["main_min"],br["main_max"]); main_ax=ax_bottom; tail_ax=ax_top
    else:
        ax_top.set_ylim(br["main_min"],br["main_max"]); ax_bottom.set_ylim(br["tail_min"],br["tail_max"]); main_ax=ax_top; tail_ax=ax_bottom
    ax_top.spines["bottom"].set_visible(False); ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x",bottom=False,labelbottom=False)
    ax_top.set_xlabel(""); ax_top.set_ylabel(""); ax_bottom.set_ylabel("")
    _remove_legend(tail_ax); _draw_vertical_break_marks(ax_top,ax_bottom)
    if zero_line and main_ax.get_ylim()[0]<=0<=main_ax.get_ylim()[1]: main_ax.axhline(0,lw=1,color="0.25")
    if title: fig.suptitle(title)
    if xlabel is not None: fig.supxlabel(xlabel)
    if ylabel is not None: fig.supylabel(ylabel)
    if rotation is not None: ax_bottom.tick_params(axis="x",rotation=rotation)
    out=dict(br); out["orientation"]="v"; return fig,(ax_top,ax_bottom),out
