# main.py
#
# API layer. Model gets fit once at startup and cached on app.state --
# refitting Dixon-Coles per request would mean re-running L-BFGS-B on
# every hit, which is pointless when the underlying match data only
# changes when we re-run ingest.

from contextlib import asynccontextmanager
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ledger.prediction_ledger import add_prediction, init_ledger
from models.dixon_coles import DixonColesModel
from validation.walk_forward import load_matches

DATA_DIR = Path("data/raw")
LEDGER_PATH = "predictions.duckdb"
# ledger writes are off by default -- render's disk is wiped on every redeploy and
# random visitors shouldn't be adding rows to a hash chain I want to keep clean.
# flip LOG_PREDICTIONS=1 for runs where I actually want the ledger
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "0") == "1"
MODEL_VERSION = "dixon_coles_v1"  # bump this whenever fit params/approach change, ledger tracks it per-row


def current_season_teams(data_dir: Path) -> list[str]:
    # the fit uses every season, but the model.teams list also has teams that got
    # relegated -- only the newest file tells me who's actually in the league now
    newest = sorted(data_dir.glob("E0_*.csv"))[-1]
    df = load_matches(newest)
    return sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))


def load_all_matches() -> pd.DataFrame:
    # ingest/fetch_data.py drops one CSV per season -- stitch them into
    # one chronological frame for fit(), same as walk_forward does per-window
    csv_paths = sorted(DATA_DIR.glob("E0_*.csv"))
    if not csv_paths:
        raise RuntimeError(
            f"No match data in {DATA_DIR}. Run `python -m ingest.fetch_data` first."
        )
    frames = [load_matches(path) for path in csv_paths]
    matches = pd.concat(frames, ignore_index=True)
    return matches.sort_values("Date").reset_index(drop=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if LOG_PREDICTIONS:
        init_ledger(LEDGER_PATH)

    matches = load_all_matches()
    model = DixonColesModel()
    model.fit(matches)

    app.state.model = model
    app.state.model_fitted_at = datetime.now(timezone.utc).isoformat()
    app.state.training_rows = len(matches)
    app.state.current_teams = current_season_teams(DATA_DIR)
    app.state.last_match_date = matches["Date"].max().date().isoformat()

    yield  # app runs here

    # nothing to tear down -- duckdb connections are opened/closed per call in the ledger module


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str


class Scoreline(BaseModel):
    home_goals: int
    away_goals: int
    prob: float


class PredictionResponse(BaseModel):
    prediction_id: Optional[str] = None  # only set when LOG_PREDICTIONS is on
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_expected_goals: float
    away_expected_goals: float
    top_scorelines: list[Scoreline]
    model_version: str


@app.get("/health")
def health():
    # getattr because a bare TestClient (no lifespan) never sets this
    return {"status": "ok", "data_through": getattr(app.state, "last_match_date", None)}


@app.get("/teams")
def teams():
    # so callers actually know what strings /predict will accept --
    # team names have to match football-data.co.uk's naming exactly
    return {"teams": app.state.current_teams}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    model: DixonColesModel = app.state.model

    unknown = [t for t in (req.home_team, req.away_team) if t not in app.state.current_teams]
    if unknown:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team(s): {unknown}. Check /teams for valid names.",
        )
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must differ")

    outcome = model.predict_match_outcome(req.home_team, req.away_team)
    home_xg, away_xg = model.predict_expected_goals(req.home_team, req.away_team)

    prediction_id = None
    if LOG_PREDICTIONS:
        # no real fixture id system yet -- good enough to make ledger rows
        # distinguishable and traceable, revisit if I add a fixtures table
        match_id = f"{req.home_team}_vs_{req.away_team}_{datetime.now(timezone.utc).date().isoformat()}"
        prediction_id = add_prediction(
            LEDGER_PATH,
            match_id=match_id,
            home_team=req.home_team,
            away_team=req.away_team,
            model_version=MODEL_VERSION,
            home_win_prob=outcome["home_win"],
            draw_prob=outcome["draw"],
            away_win_prob=outcome["away_win"],
        )

    scorelines = [
        Scoreline(home_goals=h, away_goals=a, prob=p)
        for h, a, p in model.predict_scorelines(req.home_team, req.away_team)
    ]

    return PredictionResponse(
        prediction_id=prediction_id,
        home_team=req.home_team,
        away_team=req.away_team,
        home_win_prob=outcome["home_win"],
        draw_prob=outcome["draw"],
        away_win_prob=outcome["away_win"],
        home_expected_goals=home_xg,
        away_expected_goals=away_xg,
        top_scorelines=scorelines,
        model_version=MODEL_VERSION,
    )