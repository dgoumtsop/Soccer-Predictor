# tests/test_fixtures_and_refresh.py
#
# The sandbox/CI can't depend on football-data.co.uk being up, so requests.get
# is faked and the sample csv mimics the real fixtures.csv layout (all leagues
# in one file, dd/mm/yyyy dates).

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from fixture_list import load_fixtures, upcoming
from ingest import fetch_fixtures, refresh

SAMPLE = (
    "Div,Date,Time,HomeTeam,AwayTeam,B365H\n"
    "E0,03/10/2026,15:00,Arsenal,Leeds,1.30\n"
    "E0,04/10/2026,16:30,Chelsea,Everton,1.80\n"
    "E0,25/10/2026,15:00,Fulham,Arsenal,4.00\n"
    "E1,03/10/2026,12:30,Norwich,Hull,2.10\n"
    "SP1,03/10/2026,20:00,Barcelona,Sevilla,1.50\n"
)


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def test_parse_fixtures_keeps_only_premier_league_and_iso_dates():
    df = fetch_fixtures.parse_fixtures(SAMPLE.encode())
    assert list(df.columns) == ["Date", "Time", "HomeTeam", "AwayTeam"]
    assert len(df) == 3
    assert df.iloc[0].to_dict() == {
        "Date": "2026-10-03", "Time": "15:00", "HomeTeam": "Arsenal", "AwayTeam": "Leeds",
    }


def test_parse_fixtures_rejects_html_error_page():
    with pytest.raises(ValueError):
        fetch_fixtures.parse_fixtures(b"Div,Date\n<html>not found</html>,x\n")


def test_upcoming_window_and_ordering(tmp_path):
    path = tmp_path / "E0.csv"
    fetch_fixtures.parse_fixtures(SAMPLE.encode()).to_csv(path, index=False)
    df = load_fixtures(path)

    week = upcoming(df, date(2026, 10, 2), days=7)
    assert [(f["home_team"], f["away_team"]) for f in week] == [("Arsenal", "Leeds"), ("Chelsea", "Everton")]
    assert week[0]["date"] == "2026-10-03" and week[0]["time"] == "15:00"

    # today itself counts, yesterday's games don't
    assert len(upcoming(df, date(2026, 10, 3), days=1)) == 1
    assert upcoming(df, date(2026, 10, 5), days=7) == []


def test_load_fixtures_missing_file_is_empty(tmp_path):
    assert load_fixtures(tmp_path / "nope.csv").empty


def test_refresh_downloads_fixtures_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_fixtures.requests, "get", lambda *a, **k: FakeResponse(SAMPLE.encode()))
    out = tmp_path / "fixtures" / "E0.csv"
    assert fetch_fixtures.download_fixtures(out) == 3
    assert out.exists()


RESULTS_3 = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2026,Arsenal,Chelsea,2,1\n08/08/2026,Everton,Fulham,0,0\n15/08/2026,Leeds,Arsenal,1,3\n"
RESULTS_2 = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2026,Arsenal,Chelsea,2,1\n08/08/2026,Everton,Fulham,0,0\n"
RESULTS_4 = RESULTS_3 + "22/08/2026,Chelsea,Leeds,1,1\n"


def _fake_results(monkeypatch, text):
    monkeypatch.setattr(refresh.requests, "get", lambda *a, **k: FakeResponse(text.encode()))


def test_refresh_results_updates_when_file_grows(tmp_path, monkeypatch):
    (tmp_path / "E0_2627.csv").write_bytes(RESULTS_3.encode())
    _fake_results(monkeypatch, RESULTS_4)
    msg = refresh.refresh_results("2627", tmp_path)
    assert "3 -> 4" in msg
    assert (tmp_path / "E0_2627.csv").read_text() == RESULTS_4


def test_refresh_results_keeps_old_file_when_download_shrinks(tmp_path, monkeypatch):
    (tmp_path / "E0_2627.csv").write_bytes(RESULTS_3.encode())
    _fake_results(monkeypatch, RESULTS_2)
    with pytest.raises(ValueError):
        refresh.refresh_results("2627", tmp_path)
    assert (tmp_path / "E0_2627.csv").read_text() == RESULTS_3


def test_refresh_results_rejects_garbage_and_reports_no_change(tmp_path, monkeypatch):
    (tmp_path / "E0_2627.csv").write_bytes(RESULTS_3.encode())
    _fake_results(monkeypatch, "<html>oops</html>")
    with pytest.raises(Exception):
        refresh.refresh_results("2627", tmp_path)
    assert (tmp_path / "E0_2627.csv").read_text() == RESULTS_3

    _fake_results(monkeypatch, RESULTS_3)
    assert "no change" in refresh.refresh_results("2627", tmp_path)


def test_refresh_main_runs_both_steps_even_if_one_fails(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(refresh, "refresh_results", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(refresh, "download_fixtures", lambda: calls.append("fixtures") or 5)
    assert refresh.main() == 1
    assert calls == ["fixtures"]


# ---- api ----

def _write_season(path):
    from tests.test_main_api import _write_synthetic_season
    _write_synthetic_season(path)


@pytest.fixture
def fixtures_client(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    _write_season(data_dir / "E0_2425.csv")

    fixtures_path = tmp_path / "E0.csv"
    # Hull isn't in the synthetic season, like a freshly promoted team
    text = SAMPLE.replace("Leeds", "Hull")
    fetch_fixtures.parse_fixtures(text.encode()).to_csv(fixtures_path, index=False)

    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "FIXTURES_PATH", fixtures_path)
    monkeypatch.setattr(main, "LEDGER_PATH", str(tmp_path / "p.duckdb"))
    monkeypatch.setattr(main, "_today", lambda: date(2026, 10, 2))
    with TestClient(main.app) as client:
        yield client


def test_fixtures_endpoint_flags_unpredictable_teams(fixtures_client):
    body = fixtures_client.get("/fixtures/upcoming?days=7").json()["fixtures"]
    assert [(f["home_team"], f["away_team"], f["predictable"]) for f in body] == [
        ("Arsenal", "Hull", False),
        ("Chelsea", "Everton", True),
    ]


def test_fixtures_endpoint_validates_days(fixtures_client):
    assert fixtures_client.get("/fixtures/upcoming?days=0").status_code == 422
    assert fixtures_client.get("/fixtures/upcoming?days=999").status_code == 422


def test_api_boots_without_fixtures_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    _write_season(data_dir / "E0_2425.csv")
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "FIXTURES_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(main, "LEDGER_PATH", str(tmp_path / "p.duckdb"))
    with TestClient(main.app) as client:
        assert client.get("/fixtures/upcoming").json() == {"fixtures": []}