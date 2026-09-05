"""
Home/away split form: same idea as team_form.py, but a team's home form
is computed only from their own past home matches, and away form only
from their own past away matches. Interleaved matches from the other
venue don't count toward either -- so a team's 5th home match might be
their 9th match overall, and that's expected.

Reuses the long-format + shift(1) leakage-guard pattern from
team_form.py, just with an extra "venue" key to group on before
rolling.
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
    Same split-into-2-rows idea as team_form.py, but each row also
    tags which venue that appearance was for the team ("home" or
    "away"), so home and away history can be rolled separately.
    """
    home_rows = pd.DataFrame({
        "team": matches["HomeTeam"].values,
        "venue": "home",
        "date": matches["Date"].values,
        "goals_for": matches["FTHG"].values,
        "goals_against": matches["FTAG"].values,
        "match_index": matches.index,
    })
    away_rows = pd.DataFrame({
        "team": matches["AwayTeam"].values,
        "venue": "away",
        "date": matches["Date"].values,
        "goals_for": matches["FTAG"].values,
        "goals_against": matches["FTHG"].values,
        "match_index": matches.index,
    })

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df["points"] = [
        _match_points(gf, ga) for gf, ga in zip(long_df["goals_for"], long_df["goals_against"])
    ]
    # sort within (team, venue) by date so shift(1)/rolling only ever
    # looks back within that team's history at that venue
    long_df = long_df.sort_values(["team", "venue", "date"]).reset_index(drop=True)
    return long_df


def add_home_away_form_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Adds home_form_points, home_form_goals_for, home_form_goals_against,
    home_form_matches_played (and the away_ equivalents) to a copy of
    matches.

    Unlike team_form.add_form_features, "home form" here is computed
    strictly from the team's past matches *as the home side* -- their
    away matches never enter the window, and vice versa. Same
    shift(1) leakage guard: a match's own result is excluded from its
    own form calc.

    Teams with fewer than `window` prior matches at that venue just
    get whatever venue-specific history exists -- could be 0, even if
    they have plenty of overall history at the other venue.
    """
    matches = matches.reset_index(drop=True)
    long_df = _to_long_format(matches)

    grouped = long_df.groupby(["team", "venue"])
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

    form_cols = ["match_index", "team", "venue", "form_points", "form_goals_for",
                 "form_goals_against", "form_matches_played"]
    form_lookup = long_df[form_cols]

    result = matches.copy()
    result["match_index"] = result.index

    # home side: only take rows where venue == "home" for that team
    home_lookup = form_lookup[form_lookup["venue"] == "home"].drop(columns="venue")
    result = result.merge(
        home_lookup, left_on=["match_index", "HomeTeam"],
        right_on=["match_index", "team"], how="left"
    ).drop(columns="team").rename(columns={
        "form_points": "home_form_points",
        "form_goals_for": "home_form_goals_for",
        "form_goals_against": "home_form_goals_against",
        "form_matches_played": "home_form_matches_played",
    })

    # away side: only take rows where venue == "away" for that team
    away_lookup = form_lookup[form_lookup["venue"] == "away"].drop(columns="venue")
    result = result.merge(
        away_lookup, left_on=["match_index", "AwayTeam"],
        right_on=["match_index", "team"], how="left"
    ).drop(columns=["team", "match_index"]).rename(columns={
        "form_points": "away_form_points",
        "form_goals_for": "away_form_goals_for",
        "form_goals_against": "away_form_goals_against",
        "form_matches_played": "away_form_matches_played",
    })

    # a team's very first appearance at a given venue has no prior
    # history there -- fill with 0 rather than leaving NaN, matching
    # team_form.py's convention for matches_played.
    for col in ["home_form_points", "home_form_goals_for", "home_form_goals_against",
                "home_form_matches_played", "away_form_points", "away_form_goals_for",
                "away_form_goals_against", "away_form_matches_played"]:
        result[col] = result[col].fillna(0)

    return result