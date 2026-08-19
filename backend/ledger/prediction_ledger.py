"""
Append-only prediction ledger.

Every prediction the model makes gets written here, once, with a hash
that chains to the previous record. If anyone edits an old row after
the fact, recomputing the chain will no longer match what's stored —
that's what makes this tamper-evident rather than just "a table we
promise not to delete from."

Important design point: the hash covers only what's known AT PREDICTION
TIME (teams, probabilities, model version, timestamp, previous hash).
`actual_result` is filled in later, once the real match has been played —
it is deliberately NOT part of the hash, because it genuinely isn't known
yet when the prediction is made. Hashing it would be pretending we knew
something we didn't.
"""

import hashlib
import uuid
from datetime import datetime, timezone

import duckdb

GENESIS_HASH = "0" * 64  # what the "previous hash" is for the very first row


def init_ledger(db_path: str) -> None:
    """Create the predictions table if it doesn't already exist."""
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id   VARCHAR PRIMARY KEY,
            match_id        VARCHAR,
            home_team       VARCHAR,
            away_team       VARCHAR,
            model_version   VARCHAR,
            predicted_at    VARCHAR,
            home_win_prob   DOUBLE,
            draw_prob       DOUBLE,
            away_win_prob   DOUBLE,
            actual_result   VARCHAR,
            record_hash     VARCHAR,
            sequence_num    BIGINT
        )
    """)
    con.close()


def _get_last_record(con) -> tuple[str, int]:
    """
    Returns (previous_hash, previous_sequence_num) for whatever the most
    recently written row is. If the ledger is empty, returns the genesis
    hash and sequence 0, so the very first row has something to chain to.
    """
    result = con.execute(
        "SELECT record_hash, sequence_num FROM predictions "
        "ORDER BY sequence_num DESC LIMIT 1"
    ).fetchone()

    if result is None:
        return GENESIS_HASH, 0

    return result[0], result[1]


def _compute_hash(
    prediction_id: str,
    match_id: str,
    home_team: str,
    away_team: str,
    model_version: str,
    predicted_at: str,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    previous_hash: str,
) -> str:
    """
    One string built from every field that's known at prediction time,
    plus the previous record's hash. Hash that string with SHA-256.

    Chaining to the previous hash is what makes this a *chain* rather than
    a pile of independently-hashed rows — changing any past row changes
    its hash, which no longer matches what the next row chained to, which
    breaks everything after it. That downstream breakage is the whole
    point: it makes tampering detectable, not just theoretically possible
    to notice.
    """
    payload = "|".join([
        prediction_id,
        match_id,
        home_team,
        away_team,
        model_version,
        predicted_at,
        f"{home_win_prob:.10f}",
        f"{draw_prob:.10f}",
        f"{away_win_prob:.10f}",
        previous_hash,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_prediction(
    db_path: str,
    match_id: str,
    home_team: str,
    away_team: str,
    model_version: str,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
) -> str:
    """
    Write one new prediction to the ledger. Returns the prediction_id.
    """
    con = duckdb.connect(db_path)

    prediction_id = str(uuid.uuid4())
    predicted_at = datetime.now(timezone.utc).isoformat()
    previous_hash, previous_seq = _get_last_record(con)

    record_hash = _compute_hash(
        prediction_id, match_id, home_team, away_team, model_version,
        predicted_at, home_win_prob, draw_prob, away_win_prob, previous_hash,
    )

    con.execute(
        """
        INSERT INTO predictions (
            prediction_id, match_id, home_team, away_team, model_version,
            predicted_at, home_win_prob, draw_prob, away_win_prob,
            actual_result, record_hash, sequence_num
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        [
            prediction_id, match_id, home_team, away_team, model_version,
            predicted_at, home_win_prob, draw_prob, away_win_prob,
            record_hash, previous_seq + 1,
        ],
    )

    con.close()
    return prediction_id


def record_actual_result(db_path: str, prediction_id: str, actual_result: str) -> None:
    """
    Fill in what really happened, once the match has been played.
    This deliberately does NOT touch record_hash — see module docstring.
    """
    if actual_result not in {"home_win", "draw", "away_win"}:
        raise ValueError(f"Invalid actual_result: {actual_result}")

    con = duckdb.connect(db_path)
    con.execute(
        "UPDATE predictions SET actual_result = ? WHERE prediction_id = ?",
        [actual_result, prediction_id],
    )
    con.close()


def verify_chain(db_path: str) -> dict:
    """
    Walk every row in sequence order and recompute each hash from scratch,
    checking it against what's stored. If any row's stored hash doesn't
    match what we recompute, the chain is broken from that point on —
    either that row or something before it was tampered with.

    Returns a dict describing whether the chain is intact, and if not,
    at which row it broke.
    """
    con = duckdb.connect(db_path)
    rows = con.execute(
        """
        SELECT prediction_id, match_id, home_team, away_team, model_version,
               predicted_at, home_win_prob, draw_prob, away_win_prob,
               record_hash, sequence_num
        FROM predictions ORDER BY sequence_num ASC
        """
    ).fetchall()
    con.close()

    previous_hash = GENESIS_HASH

    for row in rows:
        (prediction_id, match_id, home_team, away_team, model_version,
         predicted_at, home_win_prob, draw_prob, away_win_prob,
         stored_hash, sequence_num) = row

        recomputed_hash = _compute_hash(
            prediction_id, match_id, home_team, away_team, model_version,
            predicted_at, home_win_prob, draw_prob, away_win_prob,
            previous_hash,
        )

        if recomputed_hash != stored_hash:
            return {
                "valid": False,
                "broken_at_sequence": sequence_num,
                "broken_at_prediction_id": prediction_id,
            }

        previous_hash = stored_hash

    return {"valid": True, "records_checked": len(rows)}