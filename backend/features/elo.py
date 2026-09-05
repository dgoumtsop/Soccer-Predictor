"""
Elo ratings. Can't vectorize this one like team_form/home_away_form 
each team's rating after a match depends on their rating going into
that match, which depends on the one before that, so I'm looping
through matches in date order and updating a running rating dict as
I go.

Home advantage: I add a flat bonus to the home team's rating before
computing the expected score, not after -- so it shifts the
probability, not the raw rating number itself.

Margin of victory: a bigger win is somewhat stronger evidence, but I
don't want a 6-0 blowing everything else out of the water, so I'm
using the standard World Football Elo dampened multiplier -- it
flattens out hard past a 3+ goal margin.
"""

import pandas as pd


def expected_score(home_rating: float, away_rating: float, home_advantage: float = 100) -> tuple:
    """
    Standard Elo logistic, just with the home bump folded into the
    home team's rating before I take the diff.
    """
    adjusted_home_rating = home_rating + home_advantage
    diff = adjusted_home_rating - away_rating
    expected_home = 1 / (1 + 10 ** (-diff / 400))
    expected_away = 1 - expected_home
    return expected_home, expected_away


def goal_diff_multiplier(goal_diff: int) -> float:
    """
    Dampens how much a blowout matters. 0/1 goal margin = no extra
    weight, then it grows but flattens out fast -- by the time you're
    at 3+ goals the difference between say a 4-0 and a 5-0 barely
    moves this number.
    """
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def add_elo_features(matches: pd.DataFrame, initial_rating: float = 1500.0,
                      k: float = 20, home_advantage: float = 100) -> pd.DataFrame:
    """
    Adds home_elo_pre and away_elo_pre -- each team's rating right
    before that match, i.e. before this match's own result could
    possibly affect it. That's the leakage guard here, same spirit as
    the shift(1) in the form modules, just done manually since I'm
    looping instead of rolling.

    Sorting by Date first since the whole thing falls apart if matches
    aren't processed in chronological order.
    """
    matches = matches.sort_values("Date").reset_index(drop=True)
    ratings = {}
    home_elo_pre = []
    away_elo_pre = []

    for _, row in matches.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        # grab whatever rating each team is walking in with -- brand
        # new teams just start at initial_rating
        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)

        home_elo_pre.append(home_rating)
        away_elo_pre.append(away_rating)

        home_goals = row["FTHG"]
        away_goals = row["FTAG"]
        goal_diff = home_goals - away_goals

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals == away_goals:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        expected_home, expected_away = expected_score(home_rating, away_rating, home_advantage)
        mult = goal_diff_multiplier(goal_diff)

        # now I can actually update both ratings for next time
        ratings[home_team] = home_rating + k * mult * (actual_home - expected_home)
        ratings[away_team] = away_rating + k * mult * (actual_away - expected_away)

    result = matches.copy()
    result["home_elo_pre"] = home_elo_pre
    result["away_elo_pre"] = away_elo_pre
    return result