# tests/test_main_api.py
#
# Synthetic match data + tmp paths so this doesn't depend on real
# football-data.co.uk downloads or a shared ledger file -- same idea
# as the synthetic fixture in test_dixon_coles.py's time-decay test,
# just enough teams/games for fit() to converge cleanly.

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main


def _write_synthetic_season(csv_path):
    np.random.seed(0)
    teams = ["Arsenal", "Chelsea", "Everton", "Fulham"]
    start = pd.Timestamp("2024-08-01")
    rows = []
    game = 0
    # round robin, home and away, a few times over so fit has enough signal
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                rows.append({
                    "Date": (start + pd.Timedelta(days=game)).strftime("%d/%m/%Y"),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": np.random.poisson(1.4),
                    "FTAG": np.random.poisson(1.1),
                })
                game += 1
    pd.DataFrame(rows).to_csv(csv_path, index=False)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    _write_synthetic_season(data_dir / "E0_2425.csv")

    ledger_path = str(tmp_path / "predictions.duckdb")

    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "LEDGER_PATH", ledger_path)

    # lifespan only runs inside the `with` block -- this is what actually
    # fits the model and initializes the ledger against our tmp paths
    with TestClient(main.app) as client:
        yield client


def test_health_check(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["data_through"]


def test_teams_lists_fitted_teams(api_client):
    response = api_client.get("/teams")
    assert response.status_code == 200
    teams = response.json()["teams"]
    assert set(teams) == {"Arsenal", "Chelsea", "Everton", "Fulham"}


def test_predict_returns_probabilities(api_client):
    response = api_client.post(
        "/predict", json={"home_team": "Arsenal", "away_team": "Chelsea"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["prediction_id"] is None  # ledger off by default
    assert body["model_version"] == main.MODEL_VERSION

    probs = [body["home_win_prob"], body["draw_prob"], body["away_win_prob"]]
    assert abs(sum(probs) - 1.0) < 0.01
    assert all(0 <= p <= 1 for p in probs)
    assert body["home_expected_goals"] > 0
    assert body["away_expected_goals"] > 0

    scores = body["top_scorelines"]
    assert len(scores) == 3
    assert scores[0]["prob"] >= scores[1]["prob"] >= scores[2]["prob"]


def test_predict_writes_to_ledger_when_enabled(api_client, monkeypatch):
    monkeypatch.setattr(main, "LOG_PREDICTIONS", True)
    from ledger.prediction_ledger import init_ledger
    init_ledger(main.LEDGER_PATH)
    api_client.post("/predict", json={"home_team": "Arsenal", "away_team": "Chelsea"})

    from ledger.prediction_ledger import verify_chain

    result = verify_chain(main.LEDGER_PATH)
    assert result["valid"] is True
    assert result["records_checked"] == 1


def test_predict_unknown_team_returns_404(api_client):
    response = api_client.post(
        "/predict", json={"home_team": "Arsenal", "away_team": "Nonexistent FC"}
    )
    assert response.status_code == 404


def test_predict_same_team_twice_returns_400(api_client):
    response = api_client.post(
        "/predict", json={"home_team": "Arsenal", "away_team": "Arsenal"}
    )
    assert response.status_code == 400


def test_teams_only_lists_current_season(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    _write_synthetic_season(data_dir / "E0_2425.csv")
    # newest season is missing Fulham, like a relegated team
    df = pd.read_csv(data_dir / "E0_2425.csv")
    df = df[(df.HomeTeam != "Fulham") & (df.AwayTeam != "Fulham")]
    df["Date"] = (pd.to_datetime(df["Date"], dayfirst=True) + pd.Timedelta(days=400)).dt.strftime("%d/%m/%Y")
    df.to_csv(data_dir / "E0_2526.csv", index=False)

    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "LEDGER_PATH", str(tmp_path / "p.duckdb"))
    with TestClient(main.app) as client:
        assert set(client.get("/teams").json()["teams"]) == {"Arsenal", "Chelsea", "Everton"}
        r = client.post("/predict", json={"home_team": "Arsenal", "away_team": "Fulham"})
        assert r.status_code == 404