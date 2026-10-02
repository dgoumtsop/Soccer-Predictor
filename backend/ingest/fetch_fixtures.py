# ingest/fetch_fixtures.py
#
# football-data.co.uk keeps one fixtures.csv with the upcoming round(s) for
# every league. I only want the Premier League rows (Div == E0), and I want
# the dates normalized to ISO so nothing downstream has to guess dd/mm vs mm/dd.

import io
from pathlib import Path

import pandas as pd
import requests

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_PATH = Path("data/fixtures/E0.csv")


def parse_fixtures(content: bytes) -> pd.DataFrame:
    # their csvs aren't always utf-8, and latin-1 can't fail to decode
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    df = pd.read_csv(io.StringIO(text))

    missing = {"Div", "Date", "HomeTeam", "AwayTeam"} - set(df.columns)
    if missing:
        # if the site hands back an html error page I want to blow up here,
        # not quietly overwrite good fixtures with nothing
        raise ValueError(f"fixtures.csv is missing columns: {sorted(missing)}")

    df = df[df["Div"] == "E0"].copy()
    if "Time" not in df.columns:
        df["Time"] = ""

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Time"] = df["Time"].fillna("")

    return df[["Date", "Time", "HomeTeam", "AwayTeam"]].reset_index(drop=True)


def download_fixtures(path: Path = FIXTURES_PATH) -> int:
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    df = parse_fixtures(response.content)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} fixtures to {path}")
    return len(df)


if __name__ == "__main__":
    download_fixtures()