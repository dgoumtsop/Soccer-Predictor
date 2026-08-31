import pandas as pd
from features.team_form import add_form_features


def test_first_match_has_no_prior_form():
    """A team's very first appearance in the data has no history --
    form should show 0 matches played, not some default/leaked value."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 2, "FTAG": 1},
    ])
    result = add_form_features(matches, window=5)
    assert result.loc[0, "home_form_matches_played"] == 0
    assert result.loc[0, "away_form_matches_played"] == 0


def test_form_excludes_current_match():
    """The whole point of the leakage guard: a match's own result must
    never appear in its own form calculation."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamX", "FTHG": 3, "FTAG": 0},
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamY",
         "AwayTeam": "TeamA", "FTHG": 1, "FTAG": 0},
        {"Date": pd.Timestamp("2024-08-15"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamZ", "FTHG": 1, "FTAG": 1},
    ])
    result = add_form_features(matches, window=5)

    row = result.iloc[2]
    # form going into match 3 = average of match 1 (3 pts) and match 2 (0 pts)
    assert row["home_form_points"] == 1.5
    assert row["home_form_goals_for"] == 1.5
    assert row["home_form_goals_against"] == 0.5
    assert row["home_form_matches_played"] == 2


def test_window_limits_how_far_back_form_looks():
    """window=2 should only average the last 2 matches, not all history."""
    rows = []
    date = pd.Timestamp("2024-08-01")
    # TeamA plays 4 matches, always at home vs a throwaway opponent,
    # scoring 1, 2, 3, then we check what form looks like for a 5th match
    for i, goals in enumerate([1, 2, 3]):
        rows.append({"Date": date + pd.Timedelta(days=7 * i), "HomeTeam": "TeamA",
                     "AwayTeam": f"Opp{i}", "FTHG": goals, "FTAG": 0})
    rows.append({"Date": date + pd.Timedelta(days=21), "HomeTeam": "TeamA",
                "AwayTeam": "Opp3", "FTHG": 0, "FTAG": 0})

    matches = pd.DataFrame(rows)
    result = add_form_features(matches, window=2)

    # form going into match 4 should only average matches 2 and 3
    # (goals 2 and 3), not match 1 (goals 1)
    row = result.iloc[3]
    assert row["home_form_goals_for"] == 2.5  # (2+3)/2
    assert row["home_form_matches_played"] == 2


def test_away_team_form_computed_independently():
    """Home and away form columns should track each side correctly,
    not accidentally swap or duplicate."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 5, "FTAG": 0},
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamC",
         "AwayTeam": "TeamB", "FTHG": 0, "FTAG": 0},
    ])
    result = add_form_features(matches, window=5)

    # TeamB's form going into match 2 = their result from match 1
    # (lost 0-5 as away team: 0 points, 0 GF, 5 GA)
    row = result.iloc[1]
    assert row["away_form_points"] == 0
    assert row["away_form_goals_for"] == 0
    assert row["away_form_goals_against"] == 5