import subprocess
import sys

requirements = """
pandas
numpy
lightgbm
scikit-learn
matplotlib
seaborn
"""

def install_requirements(package_name):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name], 
        )
    except subprocess.CalledProcessError:
        print("Failed to install required packages.")
        sys.exit(1)

for package in requirements.split("\n"):
    if package:
        install_requirements(package)

import lightgbm as lgb
import pandas as pd

def enforce_valid_odds(odds):
    if odds > -100 and odds < 100:
        return 100 if odds > 0 else -100
    return odds

def makeSpreads(point_spread, league_id):
    # Paths to the saved models
    home_model_path = 'models/model_home.txt'
    away_model_path = 'models/model_away.txt'
    
    # Load the models
    model_home = lgb.Booster(model_file=home_model_path)
    model_away = lgb.Booster(model_file=away_model_path)

    input_df = pd.DataFrame({
        'point_spread_us': [point_spread],
        'league_id': [league_id]
    })
    
    predicted_home_us = model_home.predict(input_df)[0]
    predicted_away_us = model_away.predict(input_df)[0]
    
    predicted_home_us = enforce_valid_odds(predicted_home_us)
    predicted_away_us = enforce_valid_odds(predicted_away_us)
    
    return (round(predicted_home_us), round(predicted_away_us))

def main():
    # Example prediction
    point_spread = 477.5
    league_id = 1
    odds = makeSpreads(point_spread, league_id)
    print(f"Predicted Odds for Point Spread {point_spread} and League ID {league_id}:")
    print(odds)

if __name__ == "__main__":
    main()