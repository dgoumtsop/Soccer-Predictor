
from ingest.fetch_data import build_url

def test_build_url_inserts_season_code():
    assert build_url("2526") == (
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    )
    