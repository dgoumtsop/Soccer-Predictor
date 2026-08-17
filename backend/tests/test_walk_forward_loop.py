import numpy as np
import pandas as pd
import pytest

from validation.walk_forward import (
    generate_cutoffs,
    evaluate_window,
    run_walk_forward,
)
from models.dixon_coles import DixonColesModel


def _synthetic_matches(n_teams=10, n_rounds=40, seed=0):
    np.random.seed(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    rows = []
    date = pd.Timestamp("2023-08-01")
    for _ in range(n_rounds):
        date += pd.Timedelta(days=5)
        shuffled = teams[:]
        np.random.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            rows.append({
                "Date": date,
                "HomeTeam": shuffled[i],
                "AwayTeam": shuffled[i + 1],
                "FTHG": np.random.poisson(1.4),
                "FTAG": np.random.poisson(1.1),
            })
    return pd.DataFrame(rows)


def test_generate_cutoffs_respects_min_train_days():
    matches = _synthetic_matches()
    cutoffs = generate_cutoffs(matches, step_days=30, min_train_days=90)
    assert all(c >= matches["Date"].min() + pd.Timedelta(days=90) for c in cutoffs)
    assert all(c < matches["Date"].max() for c in cutoffs)


def test_evaluate_window_skips_unknown_teams():
    matches = _synthetic_matches()
    train = matches.iloc[:50]
    model = DixonColesModel()
    model.fit(train)

    # inject a team the model never saw
    unseen_row = pd.DataFrame([{
        "Date": matches["Date"].max(),
        "HomeTeam": "GhostFC",
        "AwayTeam": train["HomeTeam"].iloc[0],
        "FTHG": 1, "FTAG": 1,
    }])
    test = pd.concat([matches.iloc[50:55], unseen_row], ignore_index=True)

    records = evaluate_window(model, test)
    skipped = [r for r in records if r["skipped"]]
    assert len(skipped) == 1
    assert skipped[0]["home_team"] == "GhostFC"


def test_run_walk_forward_produces_aggregate_row():
    matches = _synthetic_matches(n_teams=12, n_rounds=60)
    results = run_walk_forward(matches, step_days=20, min_train_days=90)

    assert "AGGREGATE" in results["cutoff"].values
    aggregate = results[results["cutoff"] == "AGGREGATE"].iloc[0]

    assert 0 <= aggregate["accuracy"] <= 1
    assert aggregate["log_loss"] > 0
    assert 0 <= aggregate["brier_score"] <= 2  # multiclass Brier upper bound is 2

    # every non-aggregate row should have evaluated at least one match
    per_window = results[results["cutoff"] != "AGGREGATE"]
    assert (per_window["matches_evaluated"] > 0).all()


def test_run_walk_forward_raises_on_insufficient_range():
    tiny = _synthetic_matches(n_teams=4, n_rounds=3)
    with pytest.raises(ValueError):
        run_walk_forward(tiny, step_days=30, min_train_days=180)