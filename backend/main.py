# main.py
#
# API layer. Model gets fit once at startup and cached on app.state --
# refitting Dixon-Coles per request would mean re-running L-BFGS-B on
# every hit, which is pointless when the underlying match data only
# changes when we re-run ingest.

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ledger.prediction_ledger import add_prediction, init_ledger
from models.dixon_coles import DixonColesModel
from validation.walk_forward import load_matches

DATA_DIR = Path("data/raw")
LEDGER_PATH = "predictions.duckdb"
MODEL_VERSION = "dixon_coles_v1"  # bump this whenever fit params/approach change, ledger tracks it per-row


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
    init_ledger(LEDGER_PATH)

    matches = load_all_matches()
    model = DixonColesModel()
    model.fit(matches)

    app.state.model = model
    app.state.model_fitted_at = datetime.now(timezone.utc).isoformat()
    app.state.training_rows = len(matches)

    yield  # app runs here

    # nothing to tear down -- duckdb connections are opened/closed per call in the ledger module


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str


class PredictionResponse(BaseModel):
    prediction_id: str
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_expected_goals: float
    away_expected_goals: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/teams")
def teams():
    # so callers actually know what strings /predict will accept --
    # team names have to match football-data.co.uk's naming exactly
    return {"teams": app.state.model.teams}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    model: DixonColesModel = app.state.model

    unknown = [t for t in (req.home_team, req.away_team) if t not in model.teams]
    if unknown:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team(s): {unknown}. Check /teams for valid names.",
        )
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must differ")

    outcome = model.predict_match_outcome(req.home_team, req.away_team)
    home_xg, away_xg = model.predict_expected_goals(req.home_team, req.away_team)

    # no real fixture id system yet -- this is good enough to make ledger rows
    # distinguishable and traceable back to what was asked, revisit if we add a fixtures table
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

    return PredictionResponse(
        prediction_id=prediction_id,
        home_team=req.home_team,
        away_team=req.away_team,
        home_win_prob=outcome["home_win"],
        draw_prob=outcome["draw"],
        away_win_prob=outcome["away_win"],
        home_expected_goals=home_xg,
        away_expected_goals=away_xg,
        model_version=MODEL_VERSION,
    )