import pandas as pd
from features.home_away_form import add_home_away_form_features


def test_first_home_appearance_has_no_prior_home_form():
    """A team's first-ever home match has no home history -- should
    show 0 matches played, not a leaked/default value."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 2, "FTAG": 1},
    ])
    result = add_home_away_form_features(matches, window=5)
    assert result.loc[0, "home_form_matches_played"] == 0
    assert result.loc[0, "away_form_matches_played"] == 0


def test_home_form_excludes_current_match():
    """Leakage guard: a match's own result must never appear in its
    own home form calculation."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamX", "FTHG": 3, "FTAG": 0},
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamY", "FTHG": 1, "FTAG": 1},
        {"Date": pd.Timestamp("2024-08-15"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamZ", "FTHG": 0, "FTAG": 0},
    ])
    result = add_home_away_form_features(matches, window=5)

    row = result.iloc[2]
    # home form going into match 3 = average of match 1 (3 pts) and
    # match 2 (1 pt), both of which were home matches for TeamA
    assert row["home_form_points"] == 2.0
    assert row["home_form_goals_for"] == 2.0
    assert row["home_form_goals_against"] == 0.5
    assert row["home_form_matches_played"] == 2


def test_interleaved_away_matches_dont_count_toward_home_form():
    """The core behavior that distinguishes this from team_form: a
    team's away matches must never enter their home form window, even
    when interleaved chronologically between home matches."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamX", "FTHG": 3, "FTAG": 0},   # TeamA home win
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamY",
         "AwayTeam": "TeamA", "FTHG": 5, "FTAG": 0},   # TeamA away loss
        {"Date": pd.Timestamp("2024-08-15"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamZ", "FTHG": 1, "FTAG": 1},   # TeamA home draw
    ])
    result = add_home_away_form_features(matches, window=5)

    row = result.iloc[2]
    # home form going into match 3 should be based ONLY on match 1
    # (TeamA's one prior home match: 3 pts), ignoring the away loss
    # in match 2 entirely
    assert row["home_form_points"] == 3.0
    assert row["home_form_goals_for"] == 3.0
    assert row["home_form_goals_against"] == 0.0
    assert row["home_form_matches_played"] == 1


def test_window_limits_how_far_back_home_form_looks():
    """window=2 should only average the last 2 home matches, not all
    home history."""
    rows = []
    date = pd.Timestamp("2024-08-01")
    for i, goals in enumerate([1, 2, 3]):
        rows.append({"Date": date + pd.Timedelta(days=7 * i), "HomeTeam": "TeamA",
                     "AwayTeam": f"Opp{i}", "FTHG": goals, "FTAG": 0})
    rows.append({"Date": date + pd.Timedelta(days=21), "HomeTeam": "TeamA",
                 "AwayTeam": "Opp3", "FTHG": 0, "FTAG": 0})

    matches = pd.DataFrame(rows)
    result = add_home_away_form_features(matches, window=2)

    row = result.iloc[3]
    assert row["home_form_goals_for"] == 2.5  # (2+3)/2, not (1+2+3)/3
    assert row["home_form_matches_played"] == 2


def test_away_form_computed_independently_from_home_form():
    """Away form columns should track a team's away-only history and
    not swap or duplicate with home form."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 5, "FTAG": 0},
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamC",
         "AwayTeam": "TeamB", "FTHG": 0, "FTAG": 0},
    ])
    result = add_home_away_form_features(matches, window=5)

    # TeamB's away form going into match 2 = their result from match 1
    # (lost 0-5 as away team: 0 points, 0 GF, 5 GA)
    row = result.iloc[1]
    assert row["away_form_points"] == 0
    assert row["away_form_goals_for"] == 0
    assert row["away_form_goals_against"] == 5