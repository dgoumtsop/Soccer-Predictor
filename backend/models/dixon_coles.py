# models/dixon_coles.py

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import pandas as pd


class DixonColesModel:
    def __init__(self):
        self.teams = None
        self.attack = {}
        self.defense = {}
        self.home_advantage = 0.0

    def fit(self, matches: pd.DataFrame, xi: float = 0.0018):
        """
        matches must have columns: HomeTeam, AwayTeam, FTHG, FTAG, Date

        xi (the Greek letter "xi") controls how fast old matches fade in
        importance. 0.0018 is the value from the original Dixon-Coles
        paper — it means a match a full year old carries roughly a third
        the weight of a match played today. Set xi=0 to disable decay
        entirely (every match weighted equally, same as before).
        """
        self.teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
        n_teams = len(self.teams)
        team_idx = {team: i for i, team in enumerate(self.teams)}

        # initial guesses: attack=1.0, defense=1.0 for every team, home_adv=0.1
        init_params = np.concatenate([
            np.ones(n_teams),      # attack ratings
            np.ones(n_teams),      # defense ratings
            [0.1]                  # home advantage
        ])

        home_idx = matches["HomeTeam"].map(team_idx).to_numpy()
        away_idx = matches["AwayTeam"].map(team_idx).to_numpy()
        home_goals = matches["FTHG"].to_numpy()
        away_goals = matches["FTAG"].to_numpy()

        # Weight each match by recency. "days_ago" = how many days before
        # the most recent match in this training set each match happened.
        # The most recent match itself gets days_ago=0, weight=1.0 (full
        # weight). Older matches get exponentially smaller weight.
        dates = matches["Date"]
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates, dayfirst=True)

        most_recent_date = dates.max()
        days_ago = (most_recent_date - dates).dt.days.to_numpy()
        weights = np.exp(-xi * days_ago)

        def neg_log_likelihood(params):
            attack = params[:n_teams]
            defense = params[n_teams:2 * n_teams]
            home_adv = params[-1]

            home_expected = attack[home_idx] * defense[away_idx] * np.exp(home_adv)
            away_expected = attack[away_idx] * defense[home_idx]

            # multiplied by its weight before summing. A match with
            # weight 0.3 only contributes 30% as much "pull" on the
            # optimizer 
            log_lik = (
                (poisson.logpmf(home_goals, home_expected) * weights).sum()
                + (poisson.logpmf(away_goals, away_expected) * weights).sum()
            )

            return -log_lik  # minimize negative log-likelihood = maximize likelihood

        result = minimize(
            neg_log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=[(0.01, None)] * (2 * n_teams) + [(None, None)]
        )

        fitted = result.x
        self.attack = {team: fitted[i] for i, team in enumerate(self.teams)}
        self.defense = {team: fitted[n_teams + i] for i, team in enumerate(self.teams)}
        self.home_advantage = fitted[-1]

    def predict_expected_goals(self, home_team: str, away_team: str):
        home_expected = (
            self.attack[home_team] * self.defense[away_team] * np.exp(self.home_advantage)
        )
        away_expected = self.attack[away_team] * self.defense[home_team]
        return home_expected, away_expected

    def predict_match_outcome(self, home_team: str, away_team: str, max_goals: int = 10):
        home_exp, away_exp = self.predict_expected_goals(home_team, away_team)

        home_win = draw = away_win = 0.0

        for h_goals in range(max_goals):
            for a_goals in range(max_goals):
                p = poisson.pmf(h_goals, home_exp) * poisson.pmf(a_goals, away_exp)
                if h_goals > a_goals:
                    home_win += p
                elif h_goals == a_goals:
                    draw += p
                else:
                    away_win += p

        return {"home_win": home_win, "draw": draw, "away_win": away_win}