import numpy as np
import scipy.optimize import minimize
import scipy.stats import poisson
import pandas as pd

class DixonColesModel:
    def __init__(self):
        self.teams = None
        self.attack = {}
        self.defense = {}
        self.home_advantage = 0.0
        
    def fit(self, matches: pd.DataFrame):
        self.teams = sorted(set(matches["Home Team"]) | set(matches["Away Team"]))
        n_teams = len(self.teams)
        team_idx = {team: i for i, team in enumerate(self.teams)}

        init_params = np.concatenate(
        np.ones(n_teams),  # attack ratings
        np.ones(n_teams),  # defense ratings
        [0.1]              # home advantage

    
    def neg_log_likelihood(params):
        attack = params[:n_teams]
        defense = params[n_teams:2 * n_teams]
        home_adv = params[-1]
        log_lik = 0.
        for _, row in matches.iterrows():
            h = team_idx[row["Home Team"]] 
            a = team_idx[row["Away Team"]]

            home_expected = attack[h] * defense[a] * np.exp(home_adv)
            away_expected = attack[a] * defense[h]

            log_lik += poisson.logpmf(row["FTHG"], home expected)
            log_lik += poisson.logpmf(row["FTAG"], away expected)

            return -log_lik
        ])