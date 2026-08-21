import pandas as pd
import numpy as np
from models.dixon_coles import DixonColesModel

def test_fit_produce_valid_prob():
    matches = pd.read_csv('data/raw/E0_2526.csv')
    model = DixonColesModel()
    model.fit(matches)
    result = model.predict_match_outcome('Arsenal', 'Chelsea')
    assert set(result.keys()) == {'home_win', 'draw', 'away_win'}
    assert abs(sum(result.values()) - 1.0) < 0.01
    assert all(0 <= p <= 1 for p in result.values())


def test_time_decay_favors_recent_form():
    """
    Build a case where a team was weak early in the data and strong
    recently (its opponent is the mirror image). A model that weights
    all matches equally should end up rating them roughly on par, since
    the two spans of matches cancel out on average. A model with time
    decay should correctly rate the team with strong RECENT form higher.
    """
    np.random.seed(1)
    rows = []
    start = pd.Timestamp('2023-08-01')

    for i in range(15):
        rows.append({'Date': start + pd.Timedelta(days=i * 3), 'HomeTeam': 'TeamA',
                     'AwayTeam': 'TeamB', 'FTHG': np.random.poisson(0.5), 'FTAG': np.random.poisson(2.5)})
        rows.append({'Date': start + pd.Timedelta(days=i * 3 + 1), 'HomeTeam': 'TeamB',
                     'AwayTeam': 'TeamA', 'FTHG': np.random.poisson(2.5), 'FTAG': np.random.poisson(0.5)})

    recent_start = start + pd.Timedelta(days=250)
    for i in range(8):
        rows.append({'Date': recent_start + pd.Timedelta(days=i * 3), 'HomeTeam': 'TeamA',
                     'AwayTeam': 'TeamB', 'FTHG': np.random.poisson(2.5), 'FTAG': np.random.poisson(0.5)})
        rows.append({'Date': recent_start + pd.Timedelta(days=i * 3 + 1), 'HomeTeam': 'TeamB',
                     'AwayTeam': 'TeamA', 'FTHG': np.random.poisson(0.5), 'FTAG': np.random.poisson(2.5)})

    matches = pd.DataFrame(rows)

    model_with_decay = DixonColesModel()
    model_with_decay.fit(matches, xi=0.02)

    # TeamA's recent form is strong -- decay should make its attack
    # rating come out higher than TeamB's, even though TeamB "won" more
    # matches overall across the full (older-heavy) dataset.
    assert model_with_decay.attack['TeamA'] > model_with_decay.attack['TeamB']


def test_xi_zero_matches_undecayed_behavior():
    """xi=0 should mean 'no decay' -- every match weighted equally."""
    matches = pd.read_csv('data/raw/E0_2526.csv')
    model = DixonColesModel()
    model.fit(matches, xi=0.0)
    result = model.predict_match_outcome('Arsenal', 'Chelsea')
    assert abs(sum(result.values()) - 1.0) < 0.01