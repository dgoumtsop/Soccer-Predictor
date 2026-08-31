"""
Team form: how each team was doing in their last N matches, using only
matches before the current one. Need the shift(1) or I'll leak the
match's own result into its own form calc.
"""

import pandas as pd


def _match_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    return 0


def _to_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Splitting each match into 2 rows (home perspective, away perspective)
    so I can group by team and roll through their matches regardless of
    whether they were home or away each time. match_index tracks which
    original match each row came from so I can merge back later.
    """
    home_rows = pd.DataFrame({
        "team": matches["HomeTeam"].values,
        "date": matches["Date"].values,
        "goals_for": matches["FTHG"].values,
        "goals_against": matches["FTAG"].values,
        "match_index": matches.index,
    })
    away_rows = pd.DataFrame({
        "team": matches["AwayTeam"].values,
        "date": matches["Date"].values,
        "goals_for": matches["FTAG"].values,
        "goals_against": matches["FTHG"].values,
        "match_index": matches.index,
    })

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df["points"] = [
        _match_points(gf, ga) for gf, ga in zip(long_df["goals_for"], long_df["goals_against"])
    ]
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)
    return long_df


def add_form_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Adds home_form_points, home_form_goals_for, home_form_goals_against,
    home_form_matches_played (and same set prefixed away_) to a copy of
    matches. Each value = that team's average over their last `window`
    matches, strictly before the current one.

    Teams with fewer than `window` prior matches (early season, newly
    promoted) just get whatever history exists -- could be 0.
    matches_played tells me how much history a value is actually
    resting on, so I'm not treating a form average from 1 match the
    same as one from 5.
    """
    matches = matches.reset_index(drop=True)
    long_df = _to_long_format(matches)

    # shift(1) = leakage guard. Excludes the current row from its own
    # rolling window so "form going in" never includes this match's
    # own result.
    grouped = long_df.groupby("team")
    long_df["form_points"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long_df["form_goals_for"] = grouped["goals_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long_df["form_goals_against"] = grouped["goals_against"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long_df["form_matches_played"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).count()
    )

    form_cols = ["match_index", "team", "form_points", "form_goals_for",
                 "form_goals_against", "form_matches_played"]
    form_lookup = long_df[form_cols]

    result = matches.copy()
    result["match_index"] = result.index

    result = result.merge(
        form_lookup, left_on=["match_index", "HomeTeam"],
        right_on=["match_index", "team"], how="left"
    ).drop(columns="team").rename(columns={
        "form_points": "home_form_points",
        "form_goals_for": "home_form_goals_for",
        "form_goals_against": "home_form_goals_against",
        "form_matches_played": "home_form_matches_played",
    })

    result = result.merge(
        form_lookup, left_on=["match_index", "AwayTeam"],
        right_on=["match_index", "team"], how="left"
    ).drop(columns=["team", "match_index"]).rename(columns={
        "form_points": "away_form_points",
        "form_goals_for": "away_form_goals_for",
        "form_goals_against": "away_form_goals_against",
        "form_matches_played": "away_form_matches_played",
    })

    return result