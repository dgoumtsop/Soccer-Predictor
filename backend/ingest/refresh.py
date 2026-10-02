# ingest/refresh.py
#
# What the daily github action runs. Only re-pulls the current season (the old
# ones never change) plus the fixtures file. If a download looks wrong I keep
# what I already have -- a bad refresh should never break a working deploy.

import io
import sys
from pathlib import Path

import pandas as pd
import requests

from ingest.fetch_data import RAW_DATA_DIR, SEASON_CODES, build_url
from ingest.fetch_fixtures import download_fixtures

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def _count_valid_rows(content: bytes) -> int:
    df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"results csv is missing columns: {missing}")
    return len(df.dropna(subset=REQUIRED))


def refresh_results(season_code: str = SEASON_CODES[-1], data_dir: Path = RAW_DATA_DIR) -> str:
    response = requests.get(build_url(season_code), timeout=30)
    response.raise_for_status()

    new_rows = _count_valid_rows(response.content)
    path = data_dir / f"E0_{season_code}.csv"
    old_rows = _count_valid_rows(path.read_bytes()) if path.exists() else 0

    # a season file only ever grows, so fewer rows means a truncated download
    if new_rows < old_rows:
        raise ValueError(f"new file has {new_rows} matches but I already have {old_rows}, keeping old")

    if path.exists() and path.read_bytes() == response.content:
        return f"{path.name}: no change ({new_rows} matches)"

    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return f"{path.name}: updated, {old_rows} -> {new_rows} matches"


def main() -> int:
    failed = False
    # run both even if the first one fails, so one flaky download doesn't block the other
    for name, step in (("results", refresh_results), ("fixtures", download_fixtures)):
        try:
            print(step() if name == "results" else f"fixtures: {step()} rows")
        except Exception as exc:
            failed = True
            print(f"{name} refresh failed: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())