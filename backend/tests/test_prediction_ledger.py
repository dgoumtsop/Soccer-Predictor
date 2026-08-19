import os
import duckdb
import pytest
 
from ledger.prediction_ledger import (
    init_ledger,
    add_prediction,
    record_actual_result,
    verify_chain,
)
 
 
@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test_ledger.duckdb")
    init_ledger(path)
    return path
 
 
def test_add_prediction_returns_id(db_path):
    pred_id = add_prediction(
        db_path, match_id="M1", home_team="Arsenal", away_team="Chelsea",
        model_version="v0.1", home_win_prob=0.5, draw_prob=0.3, away_win_prob=0.2,
    )
    assert pred_id is not None
 
 
def test_chain_is_valid_after_normal_writes(db_path):
    for i in range(5):
        add_prediction(
            db_path, match_id=f"M{i}", home_team="Arsenal", away_team="Chelsea",
            model_version="v0.1", home_win_prob=0.5, draw_prob=0.3, away_win_prob=0.2,
        )
    result = verify_chain(db_path)
    assert result["valid"] is True
    assert result["records_checked"] == 5
 
 
def test_record_actual_result_does_not_break_chain(db_path):
    pred_id = add_prediction(
        db_path, match_id="M1", home_team="Arsenal", away_team="Chelsea",
        model_version="v0.1", home_win_prob=0.5, draw_prob=0.3, away_win_prob=0.2,
    )
    record_actual_result(db_path, pred_id, "home_win")
    result = verify_chain(db_path)
    assert result["valid"] is True
 
 
def test_record_actual_result_rejects_invalid_value(db_path):
    pred_id = add_prediction(
        db_path, match_id="M1", home_team="Arsenal", away_team="Chelsea",
        model_version="v0.1", home_win_prob=0.5, draw_prob=0.3, away_win_prob=0.2,
    )
    with pytest.raises(ValueError):
        record_actual_result(db_path, pred_id, "definitely_not_a_valid_outcome")
 
 
def test_verify_chain_detects_tampering(db_path):
    """
    The whole point of the ledger. Write a few real predictions, then
    reach in and directly edit one as if someone tried to quietly change
    a past prediction after the fact. verify_chain should catch it.
    """
    for i in range(3):
        add_prediction(
            db_path, match_id=f"M{i}", home_team="Arsenal", away_team="Chelsea",
            model_version="v0.1", home_win_prob=0.5, draw_prob=0.3, away_win_prob=0.2,
        )
 
    # confirm it's valid before tampering
    assert verify_chain(db_path)["valid"] is True
 
    # simulate tampering: directly rewrite the probabilities on the
    # first row, bypassing add_prediction entirely (no re-hash happens)
    con = duckdb.connect(db_path)
    con.execute(
        "UPDATE predictions SET home_win_prob = 0.99 WHERE sequence_num = 1"
    )
    con.close()
 
    result = verify_chain(db_path)
    assert result["valid"] is False
    assert result["broken_at_sequence"] == 1
 