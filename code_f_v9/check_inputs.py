from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

EXPECTED={
    "A/q1_quality.csv":["record_id","source_set","quality_domain","fineweb_edu","ad_en"],
    "A/q1_mixture_loss.csv":["pair_id","split","model_scale_label"],
    "A/q1_domain_mapping.csv":["mixture_domain","quality_domain","mapping_type"],
    "B/q2_scaling.csv":["source_id","N_params_B","D_tokens_B","val_loss"],
    "B/q2_large_models.csv":["model_name","N_params_B","D_tokens_B"],
    "C/q3_context.csv":["model_name","max_position_embeddings"],
    "C/q4_leaderboard.csv":["model","model_key","ifeval","bbh","math_lvl5","gpqa","musr","mmlu_pro"],
    "C/q4_timeseries.csv":["model","year","average"],
    "C/q4_epoch_models.csv":["model","model_key","training_compute_flop","training_dataset_size_total"],
    "C/q4_loss_benchmark_bridge.csv":["model","val_loss","comparability_bucket"],
    "C/q4_task_metrics.csv":["model_key","task_group","task","metric","value"],
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--processed",type=Path,required=True);a=ap.parse_args();bad=0
    for rel,cols in EXPECTED.items():
        p=a.processed/rel
        if not p.exists():print(f"[MISSING] {p}");bad+=1;continue
        df=pd.read_csv(p,nrows=5);miss=[c for c in cols if c not in df.columns]
        if miss:print(f"[BAD SCHEMA] {rel}: missing {miss}");bad+=1
        else:print(f"[OK] {rel}")
    optional=a.processed/"A"/"q1_text_validation.csv"
    print(f"[{'OK' if optional.exists() else 'OPTIONAL MISSING'}] A/q1_text_validation.csv")
    if bad:raise SystemExit(f"Input check failed: {bad} required files/schema groups have problems")
    print("All required preprocessed inputs are compatible.")
if __name__=="__main__":main()
