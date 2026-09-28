from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ChunkCSVWriter, canonical_model_key, numeric, reset_dir, write_csv

BENCH_RENAME = {
    "Average ⬆️": "average",
    "IFEval": "ifeval",
    "BBH": "bbh",
    "MATH Lvl 5": "math_lvl5",
    "GPQA": "gpqa",
    "MUSR": "musr",
    "MMLU-PRO": "mmlu_pro",
}
BENCH_COLS = ["ifeval", "bbh", "math_lvl5", "gpqa", "musr", "mmlu_pro"]


def bucket_model_type(value) -> str:
    s = "" if pd.isna(value) else str(value).lower()
    if "pretrained" in s and "fine" not in s:
        return "pretrained"
    if "chat" in s or "fine-tuned" in s or "rlhf" in s or "dpo" in s or "ift" in s:
        return "posttrained"
    if "merge" in s:
        return "merge"
    if "multimodal" in s:
        return "multimodal"
    return "other"


def clean_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    x = df.rename(columns=BENCH_RENAME).copy()
    out = pd.DataFrame({
        "model": x["Model"],
        "model_key": x["Model"].map(canonical_model_key),
        "params_B": numeric(x["#Params (B)"]),
        "submission_date": pd.to_datetime(x["Submission Date"], errors="coerce"),
        "hub_license": x["Hub License"],
        "type_raw": x["Type"],
        "model_type_bucket": x["Type"].map(bucket_model_type),
        "average": numeric(x["average"]),
        "ifeval": numeric(x["ifeval"]),
        "bbh": numeric(x["bbh"]),
        "math_lvl5": numeric(x["math_lvl5"]),
        "gpqa": numeric(x["gpqa"]),
        "musr": numeric(x["musr"]),
        "mmlu_pro": numeric(x["mmlu_pro"]),
    })
    out["params"] = out["params_B"] * 1e9
    for src, dst in [
        ("Epoch_AI_Publication_Date", "epoch_publication_date"),
        ("Epoch_AI_Organization", "epoch_organization"),
        ("Epoch_AI_Open_Weights", "epoch_open_weights"),
    ]:
        if src in x.columns:
            out[dst] = pd.to_datetime(x[src], errors="coerce") if "Date" in src else x[src]
        else:
            out[dst] = np.nan
    out["benchmark_complete"] = out[BENCH_COLS].notna().all(axis=1)
    return out


def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    x = df.rename(columns={
        "Params_B": "params_B", "Average": "average", "IFEval": "ifeval", "BBH": "bbh",
        "MATH_Lvl5": "math_lvl5", "GPQA": "gpqa", "MUSR": "musr", "MMLU_PRO": "mmlu_pro",
    }).copy()
    out = pd.DataFrame({
        "model": x["Model"],
        "model_key": x["Model"].map(canonical_model_key),
        "year": numeric(x["Year"]).astype("Int64"),
        "params_B": numeric(x["params_B"]),
        "average": numeric(x["average"]),
        "ifeval": numeric(x["ifeval"]),
        "bbh": numeric(x["bbh"]),
        "math_lvl5": numeric(x["math_lvl5"]),
        "gpqa": numeric(x["gpqa"]),
        "musr": numeric(x["musr"]),
        "mmlu_pro": numeric(x["mmlu_pro"]),
        "source": x["Source"],
    })
    out["params"] = out["params_B"] * 1e9
    return out


def clean_epoch(df: pd.DataFrame) -> pd.DataFrame:
    def col(name, default=np.nan):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    out = pd.DataFrame({
        "model": col("Model"),
        "model_key": col("Model").map(canonical_model_key),
        "domain": col("Domain"),
        "task": col("Task"),
        "organization": col("Organization"),
        "publication_date": pd.to_datetime(col("Publication date"), errors="coerce"),
        "parameters": numeric(col("Parameters")),
        "training_compute_flop": numeric(col("Training compute (FLOP)")),
        "training_dataset_size_total": numeric(col("Training dataset size (total)")),
        "approach": col("Approach"),
        "confidence": col("Confidence"),
        "model_accessibility": col("Model accessibility"),
        "base_model": col("Base model"),
        "finetune_compute_flop": numeric(col("Finetune compute (FLOP)")),
        "post_training_compute_flop": numeric(col("Post-training compute (FLOP)")),
        "frontier_model": col("Frontier model"),
        "open_model_weights": col("Open model weights?"),
    })
    out["parameters_B"] = out["parameters"] / 1e9
    s = out["open_model_weights"].astype(str).str.strip().str.lower()
    out["open_weights_bool"] = s.map({"yes": True, "no": False})
    return out


def clean_bridge(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "model": df["Model"],
        "model_key": df["Model"].map(canonical_model_key),
        "N_params_B": numeric(df["N_params_B"]),
        "D_tokens_B": numeric(df["D_tokens_B"]),
        "val_loss": numeric(df["Val_Loss"]),
        "benchmark_average": numeric(df["LB_Average"]),
        "ifeval": numeric(df["LB_IFEval"]),
        "bbh": numeric(df["LB_BBH"]),
        "math_lvl5": numeric(df["LB_MATH"]),
        "gpqa": numeric(df["LB_GPQA"]),
        "musr": numeric(df["LB_MUSR"]),
        "mmlu_pro": numeric(df["LB_MMLU_PRO"]),
        "loss_source": df["Loss_Source"],
        "loss_comparability": df["Loss_Comparability"],
    })
    out["N_params"] = out["N_params_B"] * 1e9
    out["D_tokens"] = out["D_tokens_B"] * 1e9
    s = out["loss_comparability"].astype(str).str.strip().str.lower()
    out["comparability_bucket"] = np.where(
        s.str.startswith("high"), "high",
        np.where(s.str.startswith("medium"), "medium", "other")
    )
    return out


def clean_context(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "model_name": df["model_name"],
        "model_key": df["model_name"].map(canonical_model_key),
        "n_layers": numeric(df["n_layers"]),
        "n_heads": numeric(df["n_heads"]),
        "d_model": numeric(df["d_model"]),
        "vocab_size": numeric(df["vocab_size"]),
        "max_position_embeddings": numeric(df["max_position_embeddings"]),
        "training_data_TB": numeric(df["training_data_TB"]),
    })
    return out


def task_group(task: str) -> str:
    if task == "leaderboard_ifeval": return "IFEval"
    if task.startswith("leaderboard_bbh"): return "BBH"
    if task.startswith("leaderboard_math"): return "MATH"
    if task.startswith("leaderboard_gpqa"): return "GPQA"
    if task.startswith("leaderboard_musr"): return "MUSR"
    if task.startswith("leaderboard_mmlu_pro"): return "MMLU_PRO"
    if task == "leaderboard": return "leaderboard_summary"
    return "other"


def numeric_metric(v):
    if isinstance(v, bool):
        return float(v)
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def preprocess_c8(base: Path, outdir: Path, chunk_size: int = 50000):
    """Flatten the latest parseable C8 result per model into one substantive long table."""
    root = base / "detailed_results"
    writer = ChunkCSVWriter(outdir / "q4_task_metrics.csv")
    rows = []
    skipped_json = 0
    models_without_parseable_json = 0
    aggregate_tasks = {
        "leaderboard", "leaderboard_bbh", "leaderboard_math_hard", "leaderboard_gpqa",
        "leaderboard_musr", "leaderboard_mmlu_pro", "leaderboard_ifeval",
    }

    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        selected = None
        obj = None
        for p in sorted(model_dir.glob("*.json"), reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                selected = p
                break
            except Exception:
                skipped_json += 1
        if obj is None:
            models_without_parseable_json += 1
            continue

        model_name = obj.get("model_name") or model_dir.name
        model_key = canonical_model_key(model_name)
        results = obj.get("results", {}) or {}
        date_unix = numeric_metric(obj.get("date"))
        eval_date = pd.to_datetime(date_unix, unit="s", utc=True, errors="coerce") if np.isfinite(date_unix) else pd.NaT
        hib = obj.get("higher_is_better", {}) or {}
        nsamples = obj.get("n-samples", {}) or {}

        for task, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                x = numeric_metric(value)
                if not np.isfinite(x):
                    continue
                hib_val = hib.get(task, {}).get(metric) if isinstance(hib.get(task), dict) else None
                ns_val = nsamples.get(task)
                if isinstance(ns_val, dict):
                    n_original = numeric_metric(ns_val.get("original"))
                    n_effective = numeric_metric(ns_val.get("effective"))
                else:
                    n_original = numeric_metric(ns_val)
                    n_effective = numeric_metric(ns_val)

                rows.append({
                    "model_name": model_name,
                    "model_key": model_key,
                    "model_dir": model_dir.name,
                    "selected_file": selected.name,
                    "evaluation_date": eval_date,
                    "task": task,
                    "task_group": task_group(task),
                    "is_aggregate_task": task in aggregate_tasks,
                    "metric": metric,
                    "value": x,
                    "higher_is_better": hib_val,
                    "n_samples_original": n_original,
                    "n_samples_effective": n_effective,
                })
                if len(rows) >= chunk_size:
                    writer.write(pd.DataFrame(rows))
                    rows.clear()
    if rows:
        writer.write(pd.DataFrame(rows))

    print(f"C8: skipped unparsable JSON files={skipped_json}; model dirs without parseable JSON={models_without_parseable_json}")


def main():
    ap = argparse.ArgumentParser(description="Q3/Q4 preprocessing only; outputs CSV only; no PCA/GAM/bridge model is fitted.")
    ap.add_argument("--root", type=Path, required=True, help="Path to real_attachments")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = args.root / "C_efficiency_evolution"
    outdir = reset_dir(args.out / "C")

    # C2 is a strict practical replacement for C1 here because the task permits C1 or C2,
    # and C2 contains C1's leaderboard content plus Epoch matching fields.
    write_csv(clean_leaderboard(pd.read_csv(base / "leaderboard_enhanced.csv")), outdir / "q4_leaderboard.csv")
    write_csv(clean_timeseries(pd.read_csv(base / "leaderboard_extended_timeseries.csv")), outdir / "q4_timeseries.csv")
    write_csv(clean_epoch(pd.read_csv(base / "epoch_all_ai_models.csv")), outdir / "q4_epoch_models.csv")

    # C6 is the main expanded Loss-Benchmark bridge; C5 is not duplicated into another output.
    write_csv(clean_bridge(pd.read_csv(base / "loss_benchmark_bridge_expanded.csv")), outdir / "q4_loss_benchmark_bridge.csv")

    write_csv(clean_context(pd.read_csv(base / "model_architecture_metadata.csv")), outdir / "q3_context.csv")
    preprocess_c8(base, outdir)

    print(f"C preprocessing complete: {outdir}")


if __name__ == "__main__":
    main()
