from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import canonical_model_key, numeric, reset_dir, safe_log10, write_csv

SOURCE_META = {
    "B1": ("pythia_training_log_existing.csv", "real", "main_fit", "pythia"),
    "B2": ("cerebras_training_log.csv", "semi_synthetic", "family_out_validation", "cerebras"),
    "B4": ("scaling_baseline.csv", "real", "cross_family_validation", None),
    "B5": ("published_scaling_data.csv", "real", "literature_validation", None),
    "B6": ("supplementary_NQ_experiment.csv", "semi_synthetic", "quality_basic", "supplementary_NQ"),
    "B7": ("supplementary_NQ_experiment_expanded.csv", "semi_synthetic", "quality_expanded", "supplementary_NQ"),
    "B8": ("supplementary_NQ_experiment_large.csv", "semi_synthetic", "quality_large", "supplementary_NQ"),
    "B10": ("supplementary_large_baseline.csv", "estimated", "large_model_extrapolation", None),
}

CANONICAL_COLUMNS = [
    "source_id", "data_nature", "role", "source_file", "record_id", "model_family",
    "N_params_B", "D_tokens_B", "C_FLOPs_1e21", "N_params", "D_tokens", "C_FLOPs",
    "C_6ND_FLOPs", "C_to_6ND_ratio", "val_loss", "Q_score", "step", "steps",
    "is_converged", "data_type",
]


def family_key(value) -> str:
    """Normalize family names only for joining B11 metadata; no fuzzy matching."""
    if pd.isna(value):
        return ""
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cerebras": "cerebras_gpt",
        "cerebras_gpt": "cerebras_gpt",
        "pythia": "pythia",
        "olmo": "olmo",
    }
    return aliases.get(s, s)


def load_family_metadata(base: Path) -> pd.DataFrame:
    """Aggregate B11 to one row per model family for a safe many-to-one join."""
    path = base / "open_model_family_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    meta = pd.read_csv(path)
    required = {"family", "model_repo", "weight_file_count", "weight_total_bytes"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"B11 missing required columns: {sorted(missing)}")

    meta["family_key"] = meta["family"].map(family_key)
    for c in ["config_bytes", "readme_bytes", "weight_file_count", "weight_total_bytes"]:
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")

    grouped = meta.groupby("family_key", dropna=False).agg(
        family_metadata_repo_count=("model_repo", "nunique"),
        family_metadata_weight_file_count=("weight_file_count", "sum"),
        family_metadata_weight_total_bytes=("weight_total_bytes", "sum"),
        family_metadata_config_bytes_mean=("config_bytes", "mean"),
        family_metadata_readme_bytes_mean=("readme_bytes", "mean"),
    ).reset_index()
    grouped["family_metadata_available"] = True
    return grouped


def clean_source(df: pd.DataFrame, source_id: str, nature: str, role: str,
                 source_file: str, default_family: str | None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["source_id"] = source_id
    out["data_nature"] = nature
    out["role"] = role
    out["source_file"] = source_file

    if "run_id" in df.columns:
        out["record_id"] = df["run_id"].astype("string")
    elif "experiment_id" in df.columns:
        out["record_id"] = df["experiment_id"].astype("string")
    else:
        out["record_id"] = [f"{source_id}_{i}" for i in range(len(df))]

    if "family" in df.columns:
        out["model_family"] = df["family"].astype("string")
    else:
        out["model_family"] = default_family or ""

    for c in ["N_params_B", "D_tokens_B", "C_FLOPs_1e21", "val_loss", "Q_score", "step", "steps"]:
        out[c] = numeric(df[c]) if c in df.columns else np.nan

    out["is_converged"] = df["is_converged"] if "is_converged" in df.columns else np.nan
    out["data_type"] = df["data_type"].astype("string") if "data_type" in df.columns else ""

    out["N_params"] = out["N_params_B"] * 1e9
    out["D_tokens"] = out["D_tokens_B"] * 1e9
    out["C_FLOPs"] = out["C_FLOPs_1e21"] * 1e21
    out["C_6ND_FLOPs"] = 6.0 * out["N_params"] * out["D_tokens"]
    out["C_to_6ND_ratio"] = out["C_FLOPs"] / out["C_6ND_FLOPs"].replace(0, np.nan)
    return out[CANONICAL_COLUMNS]


def main():
    ap = argparse.ArgumentParser(description="Q2 preprocessing only; outputs CSV only; no scaling-law model is fitted.")
    ap.add_argument("--root", type=Path, required=True, help="Path to real_attachments")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = args.root / "B_scaling_laws"
    outdir = reset_dir(args.out / "B")
    frames = []

    for source_id, (filename, nature, role, default_family) in SOURCE_META.items():
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(clean_source(pd.read_csv(path), source_id, nature, role, filename, default_family))

    traj_frames = []
    for path in sorted((base / "training_trajectories").glob("*.csv")):
        clean = clean_source(pd.read_csv(path), "B3", "interpolated", "trajectory_validation", path.name, "pythia")
        clean["record_id"] = path.stem + "_" + clean.index.astype(str)
        traj_frames.append(clean)
    if traj_frames:
        frames.append(pd.concat(traj_frames, ignore_index=True))

    scaling = pd.concat(frames, ignore_index=True, sort=False)

    # B11: attach one-row-per-family auxiliary metadata directly to the Q2 panel.
    family_meta = load_family_metadata(base)
    scaling["family_key"] = scaling["model_family"].map(family_key)
    scaling = scaling.merge(family_meta, on="family_key", how="left", validate="many_to_one")
    scaling["family_metadata_available"] = scaling["family_metadata_available"].eq(True)
    write_csv(scaling, outdir / "q2_scaling.csv")

    b9 = pd.read_csv(base / "supplementary_large_models.csv")
    large = pd.DataFrame({
        "model_name": b9["model_name"],
        "model_key": b9["model_name"].map(canonical_model_key),
        "N_params_B": numeric(b9["N_params_B"]),
        "D_tokens_B": numeric(b9["D_tokens_B"]),
        "FLOPs": numeric(b9["FLOPs"]),
        "publication_date": pd.to_datetime(b9["publication_date"], errors="coerce"),
        "organization": b9["organization"],
        "accessibility": b9["accessibility"],
        "country": b9["country"],
    })
    large["N_params"] = large["N_params_B"] * 1e9
    large["D_tokens"] = large["D_tokens_B"] * 1e9
    large["C_6ND_FLOPs"] = 6.0 * large["N_params"] * large["D_tokens"]
    write_csv(large, outdir / "q2_large_models.csv")

    print(f"B preprocessing complete: {outdir}")


if __name__ == "__main__":
    main()
