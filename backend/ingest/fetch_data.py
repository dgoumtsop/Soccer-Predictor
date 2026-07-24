from pathlib import Path
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
SEASON_CODES = ["2223", "2324", "2425", "2526"]
RAW_DATA_DIR = Path("data/raw")

def build_url(season_code: str) -> str:
    return BASE_URL.format(season=season_code)


def download_season(season_code: str) -> None:
    url = build_url(season_code)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"E0_{season_code}.csv"
    output_path.write_bytes(response.content)
    print (f"Saved {output_path}")
if __name__ == "__main__":
    for season_code in SEASON_CODES:
        download_season(season_code)
        