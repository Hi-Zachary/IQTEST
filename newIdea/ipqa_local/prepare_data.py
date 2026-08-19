"""Prepare AGIQA-3K annotation file for IP-IQA (run on the server).

Converts the official AGIQA-3k `data.csv` (name,prompt,...,mos_quality,...,
mos_align) into the `mos_joint.xlsx` that IP-IQA's AGIQA3k dataset expects
(columns: name, prompt, mos_quality, mos_align).

Usage (from project root):
    python prepare_data.py
    # or specify paths:
    python prepare_data.py --csv <path/to/data.csv> --out <path/to/mos_joint.xlsx>
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="cache/data/aigc_qa_3k/data.csv",
        help="official AGIQA-3k data.csv",
    )
    parser.add_argument(
        "--out",
        default="cache/data/aigc_qa_3k/mos_joint.xlsx",
        help="output mos_joint.xlsx for IP-IQA",
    )
    args = parser.parse_args()

    src = Path(args.csv)
    out = Path(args.out)
    assert src.exists(), f"data.csv not found: {src}"

    df = pd.read_csv(src)
    df2 = df[["name", "prompt", "mos_quality", "mos_align"]].copy()
    df2.columns = ["name", "prompt", "mos_quality", "mos_align"]

    out.parent.mkdir(parents=True, exist_ok=True)
    df2.to_excel(out, index=False)
    print(f"saved: {out}  rows={len(df2)}")


if __name__ == "__main__":
    main()
