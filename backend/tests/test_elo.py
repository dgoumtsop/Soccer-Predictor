import pandas as pd
from features.elo import expected_score, goal_diff_multiplier, add_elo_features


def test_first_match_uses_default_initial_rating():
    """Two brand new teams should both walk in at 1500 -- nothing to
    base a different number on yet."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 1, "FTAG": 0},
    ])
    result = add_elo_features(matches)
    assert result.loc[0, "home_elo_pre"] == 1500.0
    assert result.loc[0, "away_elo_pre"] == 1500.0


def test_home_advantage_favors_home_team_at_equal_ratings():
    """With equal ratings, home advantage should push expected_home
    above 0.5 -- that's the whole point of adding it."""
    expected_home, expected_away = expected_score(1500, 1500, home_advantage=100)
    assert expected_home > 0.5
    assert expected_away < 0.5
    assert round(expected_home + expected_away, 6) == 1.0


def test_draw_between_equal_teams_shifts_ratings_toward_away_team():
    """Because home advantage makes the home team the 'expected'
    side, a draw is actually a slight letdown for them -- home rating
    should drop a bit, away rating should rise, even though the
    scoreline was even."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 1, "FTAG": 1},
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamC", "FTHG": 0, "FTAG": 0},
    ])
    result = add_elo_features(matches, initial_rating=1500, k=20, home_advantage=100)
    # TeamA's rating going into their 2nd match reflects the update
    # from the draw in match 1
    assert result.loc[1, "home_elo_pre"] < 1500.0


def test_goal_diff_multiplier_dampens_blowouts():
    """1-goal margin gets no bonus, and once you're past 3 goals the
    multiplier should barely move anymore -- a 6-0 shouldn't hit like
    6x a 1-0."""
    assert goal_diff_multiplier(0) == 1.0
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(2) == 1.5

    small_margin_step = goal_diff_multiplier(2) - goal_diff_multiplier(1)
    big_margin_step = goal_diff_multiplier(6) - goal_diff_multiplier(5)
    assert big_margin_step < small_margin_step


def test_blowout_moves_rating_more_than_narrow_win_but_not_proportionally():
    """A 4-0 should move the winner's rating more than a 1-0 would,
    but nowhere near 4x as much -- checking the dampening actually
    shows up in the rating update itself, not just the multiplier
    table."""
    narrow_win = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 1, "FTAG": 0},
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamC", "FTHG": 0, "FTAG": 0},
    ])
    blowout_win = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 4, "FTAG": 0},
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamC", "FTHG": 0, "FTAG": 0},
    ])

    narrow_result = add_elo_features(narrow_win)
    blowout_result = add_elo_features(blowout_win)

    narrow_gain = narrow_result.loc[1, "home_elo_pre"] - 1500.0
    blowout_gain = blowout_result.loc[1, "home_elo_pre"] - 1500.0

    assert blowout_gain > narrow_gain
    assert blowout_gain < narrow_gain * 4


def test_ratings_update_sequentially_not_leaking_future_matches():
    """TeamA's rating going into match 2 should reflect match 1's
    result, and their rating going into match 3 should reflect match
    2's -- never the other way around."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 3, "FTAG": 0},
        {"Date": pd.Timestamp("2024-08-08"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamC", "FTHG": 0, "FTAG": 2},
        {"Date": pd.Timestamp("2024-08-15"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamD", "FTHG": 1, "FTAG": 1},
    ])
    result = add_elo_features(matches)

    # match 1: TeamA has no history yet
    assert result.loc[0, "home_elo_pre"] == 1500.0
    # match 2: should reflect the win in match 1, so above 1500
    assert result.loc[1, "home_elo_pre"] > 1500.0
    # match 3: should reflect the loss in match 2, so lower than
    # whatever it was going into match 2 -- not still stuck at the
    # post-match-1 number
    assert result.loc[2, "home_elo_pre"] < result.loc[1, "home_elo_pre"]


def test_sorts_by_date_even_if_input_is_out_of_order():
    """If matches come in unsorted, this should still process them in
    chronological order -- otherwise the whole rating history is
    garbage."""
    matches = pd.DataFrame([
        {"Date": pd.Timestamp("2024-08-15"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamD", "FTHG": 1, "FTAG": 1},
        {"Date": pd.Timestamp("2024-08-01"), "HomeTeam": "TeamA",
         "AwayTeam": "TeamB", "FTHG": 3, "FTAG": 0},
    ])
    result = add_elo_features(matches)
    # after sorting, the 08-01 match should be first and show no
    # prior history for TeamA
    assert result.loc[0, "Date"] == pd.Timestamp("2024-08-01")
    assert result.loc[0, "home_elo_pre"] == 1500.0