from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from common.io_utils import ensure_dir, read_csv, read_json, write_csv
from common.plotting import savefig, setup_style
from q3.optimizer import PredictorEnsemble, result_row, solve_hybrid
from q3.run_optimization import derive_bounds


def _render_phase_outputs(df: pd.DataFrame, out: Path, figs: Path) -> None:
    """Create Q3 phase CSV slices and figures from an already-computed phase table."""
    required = {
        "budget",
        "L_ctx",
        "r_train",
        "r_quality",
        "r_attn",
        "Q_opt",
        "expected_loss",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "q3_budget_context_phase.csv is missing required columns: "
            + ", ".join(missing)
        )

    dd = df.copy()
    for col in [
        "budget",
        "L_ctx",
        "r_train",
        "r_quality",
        "r_attn",
        "Q_opt",
        "expected_loss",
    ]:
        dd[col] = pd.to_numeric(dd[col], errors="coerce")

    dd = dd.dropna(
        subset=[
            "budget",
            "L_ctx",
            "r_train",
            "r_quality",
            "r_attn",
            "Q_opt",
            "expected_loss",
        ]
    ).copy()
    if dd.empty:
        raise ValueError("No valid numeric rows remain for Q3 phase plotting.")

    # Round the log-budget coordinate so floating-point representation cannot create
    # near-duplicate heatmap columns (for example 19.3 vs 19.300000000000004).
    dd["log10_budget"] = np.log10(dd["budget"].to_numpy(dtype=float)).round(10)

    contexts = np.sort(dd["L_ctx"].unique())
    median_ctx = float(np.median(contexts))
    Lmain = float(contexts[np.argmin(np.abs(contexts - median_ctx))])
    scan = dd[np.isclose(dd["L_ctx"], Lmain)].copy().sort_values("budget")
    write_csv(scan, out / "q3_budget_scan.csv")

    # Continuous cost-share phase maps.  The former "dominant-cost" map
    # discarded most of the information by reducing three continuous shares to one
    # categorical winner.  Here each share is shown directly on the same 0--1 scale.
    metrics=[
        ("r_train","Training cost share"),
        ("r_quality","Quality cost share"),
        ("r_attn","Attention cost share"),
    ]
    pivots=[]
    for col,_ in metrics:
        pivots.append(
            dd.pivot_table(
                index="L_ctx",
                columns="log10_budget",
                values=col,
                aggfunc="mean",
            ).astype(float)
        )

    fig=plt.figure(figsize=(18,5),layout="constrained")
    gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,0.045])
    axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
    cbar_ax=fig.add_subplot(gs[0,3])
    crit=30000.0
    for k,((_,title),piv,ax) in enumerate(zip(metrics,pivots,axes)):
        hm=sns.heatmap(
            piv,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar=(k==2),
            cbar_ax=cbar_ax if k==2 else None,
        )
        ax.set_title(title)
        ax.set_xlabel("log10 compute budget")
        ax.set_ylabel("Context length" if k==0 else "")
        # Heatmap row centres are 0.5, 1.5, ... . Interpolate the analytic
        # L_ctx^crit=30000 position when it lies inside the observed C7 range.
        yvals=piv.index.to_numpy(dtype=float)
        if len(yvals)>=2 and yvals.min()<=crit<=yvals.max():
            ypos=float(np.interp(crit,yvals,np.arange(len(yvals),dtype=float)+0.5))
            ax.axhline(ypos,color="white",linestyle="--",linewidth=1.6)
            if k==0:
                ax.text(0.01,ypos-0.08,"Lctx crit = 30000",transform=ax.get_yaxis_transform(),color="white",fontsize=9,va="top")
    cbar_ax.set_ylabel("Cost share",rotation=90)
    fig.suptitle("Compute-cost composition over budget and context length")
    # Remove stale categorical q3_08 files from older versions so users do not
    # accidentally include the obsolete dominant-cost diagram in the paper.
    for stale in [
        figs/"q3_08_phase_dominant_cost.png",
        figs/"q3_08_phase_cost_structure.png",
    ]:
        if stale.exists():
            stale.unlink()
    savefig(figs/"q3_08_phase_cost_shares.png",fig)

    qpv = dd.pivot_table(
        index="L_ctx",
        columns="log10_budget",
        values="Q_opt",
        aggfunc="mean",
    ).astype(float)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(qpv, cmap="viridis", ax=ax)
    ax.set_title("Optimal data quality over (budget, context) phase space")
    ax.set_xlabel("log10 compute budget")
    ax.set_ylabel("Context length")
    savefig(figs / "q3_09_phase_Q.png", fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=scan, x="budget", y="expected_loss", marker="o", ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Compute budget (FLOPs)")
    ax.set_ylabel("Optimal expected loss")
    ax.set_title(f"Optimal loss vs budget at Lctx={Lmain:g}")
    savefig(figs / "q3_10_budget_loss_curve.png", fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Q3 budget scan and (C,Lctx) phase diagram")
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--q1-quality", type=Path, required=True)
    ap.add_argument("--q2", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/q3"))
    ap.add_argument("--logC-min", type=float, default=19.0)
    ap.add_argument("--logC-max", type=float, default=24.0)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument(
        "--cost-kind",
        default="exponential",
        choices=["exponential", "power", "log"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Skip all optimization and regenerate q3_budget_scan.csv plus the phase "
            "figures from an existing q3_budget_context_phase.csv in --out."
        ),
    )
    a = ap.parse_args()

    setup_style()
    out = ensure_dir(a.out)
    figs = ensure_dir(out / "figures")

    if a.plot_only:
        phase_csv = out / "q3_budget_context_phase.csv"
        if not phase_csv.exists():
            raise FileNotFoundError(
                f"--plot-only requires an existing file: {phase_csv}"
            )
        df = read_csv(phase_csv)
        _render_phase_outputs(df, out, figs)
        print(f"Q3 phase figures regenerated from existing results: {out}")
        return

    Q0 = float(read_json(a.q1_quality / "q1_quality_summary.json")["Q0_global"])
    spec = derive_bounds(a.processed, Q0)
    bcsv = a.q2 / "bootstrap" / "q2_bootstrap_scaling_params.csv"
    ens = PredictorEnsemble(
        a.q2,
        bcsv if bcsv.exists() else None,
        max_samples=30 if a.fast else 200,
    )

    ctx = read_csv(a.processed / "C" / "q3_context.csv")
    contexts = np.sort(
        pd.to_numeric(ctx.max_position_embeddings, errors="coerce").dropna().unique()
    )
    contexts = contexts[contexts > 0]
    if len(contexts) == 0:
        raise ValueError("No positive max_position_embeddings values found in q3_context.csv")
    if len(contexts) > 8:
        contexts = np.unique(np.quantile(contexts, [0, 0.2, 0.4, 0.6, 0.8, 1]).round())

    budgets = 10 ** np.arange(a.logC_min, a.logC_max + 1e-9, a.step)
    if a.fast:
        budgets = 10 ** np.array(
            [a.logC_min, (a.logC_min + a.logC_max) / 2, a.logC_max]
        )
        contexts = np.array([float(np.median(contexts))])

    rows = []
    total_cases = len(contexts) * len(budgets)
    progress = tqdm(total=total_cases, desc="Q3 budget/context phase", unit="case")
    try:
        for L in contexts:
            x0 = None
            for j, C in enumerate(budgets):
                force_de = (
                    abs(np.log10(C) - round(np.log10(C))) < 1e-8 or x0 is None
                )
                r, t = solve_hybrid(
                    ens,
                    float(C),
                    Q0,
                    float(L),
                    a.cost_kind,
                    spec,
                    True,
                    None,
                    "mean_cvar",
                    0.2,
                    a.seed + j,
                    fast=a.fast,
                    x0=None if force_de else x0,
                    show_progress=True,
                )
                rr = result_row(
                    r,
                    t,
                    ens,
                    float(C),
                    Q0,
                    float(L),
                    a.cost_kind,
                    True,
                    None,
                    "mean_cvar",
                    0.2,
                    "DE+SLSQP" if force_de else "warm_SLSQP",
                )
                rows.append(rr)
                x0 = r.x
                progress.update(1)
                progress.set_postfix(
                    Lctx=f"{L:g}", logC=f"{np.log10(C):.2f}"
                )
    finally:
        progress.close()

    df = pd.DataFrame(rows)
    write_csv(df, out / "q3_budget_context_phase.csv")
    _render_phase_outputs(df, out, figs)
    print(f"Q3 phase scan complete: {out}")


if __name__ == "__main__":
    main()
