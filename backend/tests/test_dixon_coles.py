import pandas as pd
from models.dixon_coles import DixonColesModel

def test_fit_produce_valid_prob():
    matches = pd.read_csv('data/raw/E0_2526.csv')
    model = DixonColesModel()
    model.fit(matches)
    result = model.predict_match_outcome('Arsenal', 'Chelsea')
    assert set(result.keys()) == {'home_win', 'draw', 'away_win'}
    assert abs(sum(result.values()) - 1.0) < 0.01
    assert all(0 <= p <= 1 for p in result.values())