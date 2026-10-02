# fixture_list.py
#
# Reads the cleaned fixtures file that ingest/fetch_fixtures.py writes and
# answers "what's coming up". Kept separate from main.py so I can test the
# date filtering without spinning up the whole api.

from datetime import date
from pathlib import Path

import pandas as pd

COLUMNS = ["Date", "Time", "HomeTeam", "AwayTeam"]


def load_fixtures(path: Path) -> pd.DataFrame:
    # no file just means no fixtures yet -- the api should still boot and predict
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(path)
    # fetch_fixtures already normalized these to ISO, so no dayfirst guessing here
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    return df.sort_values(["Date", "Time"]).reset_index(drop=True)


def upcoming(df: pd.DataFrame, today: date, days: int = 14) -> list[dict]:
    start = pd.Timestamp(today)
    end = start + pd.Timedelta(days=days)
    window = df[(df["Date"] >= start) & (df["Date"] < end)]

    return [
        {
            "date": row.Date.date().isoformat(),
            "time": row.Time if isinstance(row.Time, str) and row.Time else None,
            "home_team": row.HomeTeam,
            "away_team": row.AwayTeam,
        }
        for row in window.itertuples()
    ]