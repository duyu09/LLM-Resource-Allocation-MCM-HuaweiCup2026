from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

from common import ChunkCSVWriter, reset_dir, write_csv

QUALITY_FIELDS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "dsir_books", "dsir_wiki",
    "dsir_math", "qurater", "ad_en", "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_doc_unigram_entropy", "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction", "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]

LIST_FIELDS = {
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "qurater", "ad_en",
}


def scalarize(value, method: str = "mean") -> float:
    """Compress one list-valued quality signal to one scalar; no quality score Q is fitted here."""
    if value is None:
        return np.nan
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return np.nan
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan
        if method == "mean":
            return float(np.mean(arr))
        if method == "median":
            return float(np.median(arr))
        if method == "max":
            return float(np.max(arr))
        raise ValueError(f"Unsupported scalarizer: {method}")
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def quality_sources(root: Path):
    a = root / "A_data_value"
    out = [("A1", a / "slimpajama_quality_signal_sample.jsonl.xz")]
    out += [("A2", p) for p in sorted((a / "slimpajama_quality_extended").glob("arxiv_*.jsonl.xz"))]
    out += [("A3", p) for p in sorted((a / "slimpajama_quality_extended").glob("github_*.jsonl.xz"))]
    return out


def infer_quality_domain(source_set: str, obj: dict, source_path: Path) -> str:
    if source_set == "A1":
        d = obj.get("_source_domain")
        return str(d).strip().lower() if d else "unknown"
    name = source_path.name.lower()
    if "arxiv" in name:
        return "arxiv"
    if "github" in name:
        return "github"
    return "unknown"


def preprocess_quality(root: Path, outdir: Path, chunk_size: int, scalarizer: str):
    writer = ChunkCSVWriter(outdir / "q1_quality.csv")
    rows = []

    for source_set, path in quality_sources(root):
        with lzma.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rid = str(obj.get("id", ""))
                row = {
                    "record_id": rid,
                    "source_set": source_set,
                    "source_file": path.name,
                    "quality_domain": infer_quality_domain(source_set, obj, path),
                    "sub_path": obj.get("sub_path", ""),
                }
                for field in QUALITY_FIELDS:
                    row[field] = scalarize(obj.get(field), scalarizer)
                rows.append(row)
                if len(rows) >= chunk_size:
                    writer.write(pd.DataFrame(rows))
                    rows.clear()
    if rows:
        writer.write(pd.DataFrame(rows))


def clean_mixture_pair(root: Path, mix_file: str, loss_file: str, pair_id: str,
                       split: str, scale_label: str, scale_B: float,
                       mixture_nature: str, loss_nature: str) -> pd.DataFrame:
    base = root / "A_data_value" / "regmix_tables"
    mix = pd.read_csv(base / mix_file)
    loss = pd.read_csv(base / loss_file)
    if mix["index"].duplicated().any() or loss["index"].duplicated().any():
        raise ValueError(f"Duplicate index in pair {pair_id}")
    df = mix.merge(loss, on="index", how="inner", validate="one_to_one")
    if len(df) != len(mix) or len(df) != len(loss):
        raise ValueError(f"Index mismatch in pair {pair_id}")

    p_raw = [c for c in df.columns if c.startswith("train_the_pile_")]
    l_raw = [c for c in df.columns if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]
    if len(p_raw) != 17 or len(l_raw) != 13:
        raise ValueError(f"Unexpected RegMix shape in {pair_id}: p={len(p_raw)}, loss={len(l_raw)}")

    df[p_raw] = df[p_raw].apply(pd.to_numeric, errors="coerce")
    df[l_raw] = df[l_raw].apply(pd.to_numeric, errors="coerce")
    sums = df[p_raw].sum(axis=1, min_count=17)
    if (sums <= 0).fillna(True).any():
        raise ValueError(f"Invalid mixture row in {pair_id}")

    p = df[p_raw].div(sums, axis=0)
    p.columns = ["p_" + c.removeprefix("train_the_pile_") for c in p.columns]
    losses = df[l_raw].copy()
    losses.columns = ["loss_" + c.removeprefix("metric/the_pile_").removesuffix("_val_loss") for c in losses.columns]

    out = pd.concat([df[["index"]].copy(), p, losses], axis=1)
    out.insert(1, "pair_id", pair_id)
    out.insert(2, "split", split)
    out.insert(3, "model_scale_label", scale_label)
    out.insert(4, "model_scale_B", scale_B)
    out.insert(5, "mixture_nature", mixture_nature)
    out.insert(6, "loss_nature", loss_nature)
    return out


def preprocess_mixtures(root: Path, outdir: Path):
    pairs = [
        ("train_mixture_1m.csv", "train_pile_loss_1m.csv", "A4_A5", "train", "1M", 0.001, "real", "real"),
        ("test_mixture_1m.csv", "test_pile_loss_1m.csv", "A6_A7", "test", "1M", 0.001, "real", "real"),
        ("test_mixture_60m.csv", "test_pile_loss_60m.csv", "A8_A9", "test", "60M", 0.060, "real", "real"),
        ("test_mixture_1B.csv", "test_pile_loss_1B.csv", "A10_A11", "test", "1B", 1.0, "real", "real"),
        ("est_mixture_10b.csv", "est_pile_loss_10b.csv", "A12_A13", "extrapolation", "10B", 10.0, "train_subset", "extrapolated"),
        ("est_mixture_70b.csv", "est_pile_loss_70b.csv", "A14_A15", "extrapolation", "70B", 70.0, "train_subset", "extrapolated"),
    ]
    panel = pd.concat([clean_mixture_pair(root, *p) for p in pairs], ignore_index=True, sort=False)
    write_csv(panel, outdir / "q1_mixture_loss.csv")

    # A16 + A17: merge domain mapping with domain-level summary statistics.
    mapping = pd.read_csv(root / "A_data_value" / "domain_mapping_guide.csv")
    for c in ["mixture_domain", "quality_domain", "mapping_type"]:
        if c in mapping.columns:
            mapping[c] = mapping[c].astype(str).str.strip().str.lower()

    summary_path = root / "A_data_value" / "regmix_domain_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    if "domain" not in summary.columns:
        raise ValueError("A17 regmix_domain_summary.csv must contain column: domain")
    summary["domain"] = summary["domain"].astype(str).str.strip().str.lower()
    for c in ["sample_rows", "sample_bytes_requested", "avg_text_chars"]:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce")
    summary = summary.rename(columns={
        "domain": "mixture_domain",
        "source_path": "domain_source_path",
        "sample_rows": "domain_sample_rows",
        "sample_bytes_requested": "domain_sample_bytes_requested",
        "avg_text_chars": "domain_avg_text_chars",
    })
    keep = [c for c in [
        "mixture_domain", "domain_source_path", "domain_sample_rows",
        "domain_sample_bytes_requested", "domain_avg_text_chars"
    ] if c in summary.columns]
    summary = summary[keep].drop_duplicates(subset=["mixture_domain"])

    mapping = mapping.merge(summary, on="mixture_domain", how="left", validate="one_to_one")
    write_csv(mapping, outdir / "q1_domain_mapping.csv")


def preprocess_optional_a18(root: Path, outdir: Path, chunk_size: int):
    """Optional Q1 validation input: domain-level raw-text statistics from A18."""
    path = root / "A_data_value" / "regmix_domain_sample.jsonl.xz"
    if not path.exists():
        return
    writer = ChunkCSVWriter(outdir / "q1_text_validation.csv")
    rows = []
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text") or obj.get("content") or ""
            rows.append({
                "row_no": line_no,
                "domain": obj.get("_source_domain", ""),
                "text_char_count": len(text),
                "text_line_count": text.count("\n") + 1 if text else 0,
                "text_whitespace_token_count": len(text.split()) if text else 0,
            })
            if len(rows) >= chunk_size:
                writer.write(pd.DataFrame(rows))
                rows.clear()
    if rows:
        writer.write(pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser(description="Q1 preprocessing only; outputs CSV only; no formal model is fitted.")
    ap.add_argument("--root", type=Path, required=True, help="Path to real_attachments")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--list-scalarizer", choices=["mean", "median", "max"], default="mean")
    ap.add_argument("--include-a18", action="store_true")
    args = ap.parse_args()

    outdir = reset_dir(args.out / "A")
    preprocess_quality(args.root, outdir, args.chunk_size, args.list_scalarizer)
    preprocess_mixtures(args.root, outdir)
    if args.include_a18:
        preprocess_optional_a18(args.root, outdir, args.chunk_size)
    print(f"A preprocessing complete: {outdir}")


if __name__ == "__main__":
    main()
