"""Download a checksum-verified source, or explicitly use the synthetic fixture."""

import argparse
import hashlib
from http.client import IncompleteRead
import time
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
from .config import FEATURES, NUMERIC, ROOT
from .schema import Customer

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def load_data(path):
    df = pd.read_csv(path)
    missing = set(FEATURES + ["customerID", "Churn"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.customerID.isna().any() or df.customerID.duplicated().any():
        raise ValueError("customerID must be present and unique")
    if not df.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Churn must contain only Yes/No")
    for col in NUMERIC:
        blank = df[col].astype(str).str.strip().eq("")
        df[col] = pd.to_numeric(df[col].mask(blank), errors="raise")
    for row in df[FEATURES].to_dict("records"):
        if pd.isna(row["TotalCharges"]):
            row["TotalCharges"] = None
        Customer.model_validate(row)
    if df.Churn.value_counts().min() < 10 or df.Churn.nunique() != 2:
        raise ValueError("At least 10 rows of each class are required")
    return df


def download(output=ROOT / "data/raw/telco.csv"):
    # Retry interrupted transfers; never accept an unverified or partial CSV.
    for attempt in range(3):
        try:
            with urlopen(URL, timeout=45) as response:
                content = response.read()
            break
        except (OSError, IncompleteRead):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError(
            "Dataset checksum changed; inspect the source before accepting it"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_bytes(content)
    load_data(temporary)
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/telco.csv")
    args = parser.parse_args()
    try:
        print(f"Downloaded and validated: {download(args.output)}")
    except Exception as exc:
        raise SystemExit(
            f"Download failed: {exc}\nOffline smoke demo: python -m churn.train --data data/sample.csv"
        )


if __name__ == "__main__":
    main()
