from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
]

def load_matches(csv_path: str | Path) -> pd.DataFrame:
    #The returned DataFrame is sorted chronologically and contains
    #only the columns required by the prediction model.
    df = pd.read_csv(csv_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="raise",
    )

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="raise")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="raise")

    df = df.sort_values("Date").reset_index(drop=True)

    return df

def split_at_date(
    matches: pd.DataFrame,          
    cutoff_date: str,               
) -> tuple[pd.DataFrame, pd.DataFrame]:  

    #plit matches chronologically into training and test sets.

    cutoff = pd.Timestamp(cutoff_date)
    train = matches[matches["Date"] < cutoff].copy()
    test = matches[matches["Date"] >= cutoff].copy()
    if train.empty:
        raise ValueError("Training set is empty")
    if test.empty:
        raise ValueError("Test cannot be empty")
    return train, test
