from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(script: str, root: Path, out: Path, extra=None):
    cmd = [sys.executable, str(Path(__file__).with_name(script)), "--root", str(root), "--out", str(out)]
    if extra:
        cmd += extra
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Zero-step preprocessing for F problem. CSV outputs only; no formal modelling.")
    ap.add_argument("--root", type=Path, default=Path("real_attachments"), help="Path to real_attachments")
    ap.add_argument("--out", type=Path, default=Path("processed"))
    ap.add_argument("--list-scalarizer", choices=["mean", "median", "max"], default="mean")
    ap.add_argument("--include-a18", action="store_true")
    args = ap.parse_args()

    extra_a = ["--list-scalarizer", args.list_scalarizer]
    if args.include_a18:
        extra_a.append("--include-a18")

    run("preprocess_a.py", args.root, args.out, extra_a)
    run("preprocess_b.py", args.root, args.out)
    run("preprocess_c.py", args.root, args.out)
    print(f"All preprocessing complete: {args.out.resolve()}")


if __name__ == "__main__":
    main()
