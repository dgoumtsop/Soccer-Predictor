# models/dixon_coles.py

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import pandas as pd


def _tau(h_goals: int, a_goals: int, home_exp: float, away_exp: float, rho: float) -> float:
    """
    Dixon-Coles low-score correction. Only touches 0-0, 1-0, 0-1, 1-1 --
    plain Poisson treats home/away goals as independent, which underrates
    the correlation in tight low-scoring games. rho=0 disables this.
    """
    if h_goals == 0 and a_goals == 0:
        return 1 - (home_exp * away_exp * rho)
    elif h_goals == 0 and a_goals == 1:
        return 1 + (home_exp * rho)
    elif h_goals == 1 and a_goals == 0:
        return 1 + (away_exp * rho)
    elif h_goals == 1 and a_goals == 1:
        return 1 - rho
    else:
        return 1.0


class DixonColesModel:
    def __init__(self):
        self.teams = None
        self.attack = {}
        self.defense = {}
        self.home_advantage = 0.0
        self.rho = 0.0

    def fit(self, matches: pd.DataFrame, xi: float = 0.0018):
        """
        matches must have columns: HomeTeam, AwayTeam, FTHG, FTAG, Date
        xi controls time decay (0.0018 = Dixon-Coles paper default, xi=0 disables it)
        """
        self.teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
        n_teams = len(self.teams)
        team_idx = {team: i for i, team in enumerate(self.teams)}

        # initial guesses: attack=1.0, defense=1.0, home_adv=0.1, rho=0.0
        init_params = np.concatenate([
            np.ones(n_teams),
            np.ones(n_teams),
            [0.1],
            [0.0]
        ])

        home_idx = matches["HomeTeam"].map(team_idx).to_numpy()
        away_idx = matches["AwayTeam"].map(team_idx).to_numpy()
        home_goals = matches["FTHG"].to_numpy()
        away_goals = matches["FTAG"].to_numpy()

        dates = matches["Date"]
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates, dayfirst=True)

        most_recent_date = dates.max()
        days_ago = (most_recent_date - dates).dt.days.to_numpy()
        weights = np.exp(-xi * days_ago)

        def neg_log_likelihood(params):
            attack = params[:n_teams]
            defense = params[n_teams:2 * n_teams]
            home_adv = params[-2]
            rho = params[-1]

            home_expected = attack[home_idx] * defense[away_idx] * np.exp(home_adv)
            away_expected = attack[away_idx] * defense[home_idx]

            base_log_lik = (
                poisson.logpmf(home_goals, home_expected)
                + poisson.logpmf(away_goals, away_expected)
            )

            tau = np.ones_like(home_expected)

            mask_00 = (home_goals == 0) & (away_goals == 0)
            tau = np.where(mask_00, 1 - (home_expected * away_expected * rho), tau)

            mask_01 = (home_goals == 0) & (away_goals == 1)
            tau = np.where(mask_01, 1 + (home_expected * rho), tau)

            mask_10 = (home_goals == 1) & (away_goals == 0)
            tau = np.where(mask_10, 1 + (away_expected * rho), tau)

            mask_11 = (home_goals == 1) & (away_goals == 1)
            tau = np.where(mask_11, 1 - rho, tau)

            tau = np.clip(tau, 1e-10, None)  # avoid log(0)

            log_lik = ((base_log_lik + np.log(tau)) * weights).sum()

            return -log_lik

        result = minimize(
            neg_log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=[(0.01, None)] * (2 * n_teams) + [(None, None)] + [(-0.2, 0.2)]
        )

        fitted = result.x
        self.attack = {team: fitted[i] for i, team in enumerate(self.teams)}
        self.defense = {team: fitted[n_teams + i] for i, team in enumerate(self.teams)}
        self.home_advantage = fitted[-2]
        self.rho = fitted[-1]

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
                p = (
                    poisson.pmf(h_goals, home_exp)
                    * poisson.pmf(a_goals, away_exp)
                    * _tau(h_goals, a_goals, home_exp, away_exp, self.rho)
                )
                if h_goals > a_goals:
                    home_win += p
                elif h_goals == a_goals:
                    draw += p
                else:
                    away_win += p

        return {"home_win": home_win, "draw": draw, "away_win": away_win}