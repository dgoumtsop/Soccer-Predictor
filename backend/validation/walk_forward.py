from pathlib import Path
import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesModel

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

def generate_cutoffs(matches: pd.DataFrame,
                     step_days: int = 30,
                     min_train_days = 180,
) -> list[pd.Timestamp]:
    #Build a list of dates to walk forward through
    start = matches["Date"].min() + pd.Timedelta(days= min_train_days)
    end = matches["Date"].max()
    cut_offs = []
    curr = start
    while curr < end:
        cut_offs.append(curr)
        curr += pd.Timedelta(days= step_days)
        return cut_offs
    
def actual_outcome(row : pd.Series) -> str :
    if row["FTHG"] > row["FTAG"]:
        return "home_win"
    elif row["FTHG"] < row["FTAG"]:
        return "away_win"
    return "draw"

def evaluate_window(model: DixonColesModel, test: pd.DataFrame) -> list[dict]:
    known_teams = set(model.teams)
    records = []
    for _, row in test.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        if home not in known_teams or away not in known_teams:
            records.append({"skipped": True, "home_team": home, "away_team": away,})
            continue

        probs = model.predict_match_outcome(home, away)
        actual = actual_outcome(row)
        predicted = max(probs, key=probs.get)
 
        records.append({
            "skipped": False,
            "home_team": home,
            "away_team": away,
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "home_win_prob": probs["home_win"],
            "draw_prob": probs["draw"],
            "away_win_prob": probs["away_win"],
        })
 
    return records

def log_loss(records: list[dict]) -> float:
    # multiclass log loss across home/draw/away win to avoid log 0 
    eps = 1e-15
    losses = []
    for r in records:
        p = np.clip(r[f"{r['actual']}_prob"], eps, 1 - eps)
        losses.append(-np.log(p))
    return float(np.mean(losses)) if losses else float("nan")

def brier_score(records: list[dict]) -> float:
    # Brier score squared error between the predicted
 # probability vector and the one-hot actual outcome
    outcomes = ["home_win", "draw", "away_win"]
    scores = []
    for r in records:
        for outcome in outcomes:
            target = 1.0 if r["actual"] == outcome else 0.0
            scores.append((r[f"{outcome}_prob"] - target) ** 2)
    return float(np.mean(scores)) if scores else float("nan")


def _per_class_accuracy(records: list[dict], outcome_class: str) -> float | None:
    relevant = [r for r in records if r["actual"] == outcome_class]
    if not relevant:
        return None
    correct = sum(1 for r in relevant if r["correct"])
    return correct / len(relevant)
 
 
def summarize_window(records: list[dict], cutoff: pd.Timestamp) -> dict:
    scored = [r for r in records if not r["skipped"]]
    skipped = [r for r in records if r["skipped"]]
 
    if not scored:
        return {
            "cutoff": cutoff,
            "matches_evaluated": 0,
            "matches_skipped": len(skipped),
            "accuracy": None,
            "log_loss": None,
            "brier_score": None,
            "home_win_accuracy": None,
            "draw_accuracy": None,
            "away_win_accuracy": None,
        }
 
    accuracy = sum(1 for r in scored if r["correct"]) / len(scored)
 
    return {
        "cutoff": cutoff,
        "matches_evaluated": len(scored),
        "matches_skipped": len(skipped),
        "accuracy": accuracy,
        "log_loss": log_loss(scored),
        "brier_score": brier_score(scored),
        "home_win_accuracy": _per_class_accuracy(scored, "home_win"),
        "draw_accuracy": _per_class_accuracy(scored, "draw"),
        "away_win_accuracy": _per_class_accuracy(scored, "away_win"),
    }
 
 
def run_walk_forward(
    matches: pd.DataFrame,
    step_days: int = 30,
    min_train_days: int = 180,
) -> pd.DataFrame:
    """
    The core validation loop:
 
        for each rolling cutoff:
            train = matches before cutoff
            test  = matches in the following window
            fit model on train
            score model on test
            record metrics
 
    Returns one row of metrics per cutoff window, plus a final
    aggregate row averaged across all windows.
    """
    cutoffs = generate_cutoffs(matches, step_days=step_days, min_train_days=min_train_days)
 
    if not cutoffs:
        raise ValueError(
            "No valid cutoffs generated — not enough date range in `matches` "
            "for the given min_train_days/step_days."
        )
 
    window_results = []
 
    for i, cutoff in enumerate(cutoffs):
        next_cutoff = cutoffs[i + 1] if i + 1 < len(cutoffs) else matches["Date"].max() + pd.Timedelta(days=1)
 
        train = matches[matches["Date"] < cutoff]
        test = matches[(matches["Date"] >= cutoff) & (matches["Date"] < next_cutoff)]
 
        if train.empty or test.empty:
            continue
 
        model = DixonColesModel()
        model.fit(train)
 
        records = evaluate_window(model, test)
        window_results.append(summarize_window(records, cutoff))
 
    results_df = pd.DataFrame(window_results)
 
    if results_df.empty:
        raise ValueError("Walk-forward produced no scoreable windows.")
 
    # Append an aggregate row so it's trivial to compare Model A vs Model B
    # at a glance without recomputing anything downstream.
    numeric_cols = ["accuracy", "log_loss", "brier_score",
                     "home_win_accuracy", "draw_accuracy", "away_win_accuracy"]
    aggregate = {"cutoff": "AGGREGATE",
                 "matches_evaluated": results_df["matches_evaluated"].sum(),
                 "matches_skipped": results_df["matches_skipped"].sum()}
    for col in numeric_cols:
        aggregate[col] = results_df[col].mean(skipna=True)
 
    results_df = pd.concat([results_df, pd.DataFrame([aggregate])], ignore_index=True)
 
    return results_df
 
 