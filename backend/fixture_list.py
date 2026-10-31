# fixture_list.py
#
# Reads the cleaned fixtures file that ingest/fetch_fixtures.py writes and
# answers "what's coming up". Kept separate from main.py so I can test the
# date filtering without spinning up the whole api.

from datetime import date
from pathlib import Path

import pandas as pd

ODDS_COLUMNS = ["AvgH", "AvgD", "AvgA"]
COLUMNS = ["Date", "Time", "HomeTeam", "AwayTeam", *ODDS_COLUMNS]


def bookmaker_probs(home_odds, draw_odds, away_odds) -> dict | None:
    # 1/odds gives each outcome's implied chance, but the three add up to ~105%
    # because of the bookmaker's margin -- dividing by the total takes it out
    odds = [home_odds, draw_odds, away_odds]
    if any(pd.isna(o) or o <= 1 for o in odds):
        return None
    implied = [1 / o for o in odds]
    total = sum(implied)
    return {
        "home_win_prob": implied[0] / total,
        "draw_prob": implied[1] / total,
        "away_win_prob": implied[2] / total,
    }


def load_fixtures(path: Path) -> pd.DataFrame:
    # no file just means no fixtures yet -- the api should still boot and predict
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(path)
    # files written before odds were kept just won't have these columns
    for col in ODDS_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
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
            "bookmaker": bookmaker_probs(row.AvgH, row.AvgD, row.AvgA),
        }
        for row in window.itertuples()
    ]