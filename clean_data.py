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

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Function to convert American odds to Decimal odds
def american_to_decimal(american_odds):
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

# Function to convert Decimal odds to American odds
def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    elif decimal_odds > 1.0:
        return int(-100 / (decimal_odds - 1))
    else:
        # Handle cases where decimal odds are less than or equal to 1 means data is invalid
        return np.nan

# Function to enforce valid American odds
def enforce_valid_odds(odds):
    if odds > -100 and odds < 100:
        return 100 if odds > 0 else -100
    return odds

# Function to fix inconsistent point spread signs based on moneyline odds
def fix_inconsistent_pointspread(df):
    # Identify rows where pointspread_us and pointspread_eu have opposite signs
    inconsistent_mask = (df['point_spread_us'] * df['point_spread_eu']) < 0
    
    # Masks to determine if the home team is favorite or underdog based on moneyline odds
    home_fav_mask = (df['homeprice_us'] < 0) & (df['homeprice_eu'] < 2.0)
    away_fav_mask = (df['homeprice_us'] > 0) & (df['homeprice_eu'] > 2.0)
    
    # Correct point spreads for home favorites
    df.loc[inconsistent_mask & home_fav_mask, 'point_spread_us'] = abs(df.loc[inconsistent_mask & home_fav_mask, 'point_spread_us'])
    df.loc[inconsistent_mask & home_fav_mask, 'point_spread_eu'] = abs(df.loc[inconsistent_mask & home_fav_mask, 'point_spread_eu'])
    
    # Correct point spreads for away favorites
    df.loc[inconsistent_mask & away_fav_mask, 'point_spread_us'] = -abs(df.loc[inconsistent_mask & away_fav_mask, 'point_spread_us'])
    df.loc[inconsistent_mask & away_fav_mask, 'point_spread_eu'] = -abs(df.loc[inconsistent_mask & away_fav_mask, 'point_spread_eu'])
    
    # For rows that are inconsistent but do not match favorite masks, set point spreads to NaN
    other_inconsistent_mask = inconsistent_mask & ~(home_fav_mask | away_fav_mask)
    df.loc[other_inconsistent_mask, ['point_spread_us', 'point_spread_eu']] = np.nan
    
    return df

# Load and clean the data
def load_and_clean_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Replace string 'NA' with actual NaN values
    df.replace(['NA', 'na', 'NaN', 999], np.nan, inplace=True)
    
    # Convert data types
    numeric_columns = ['point_spread_us', 'point_spread_eu', 'league_id',
                       'homeprice_us', 'awayprice_us',
                       'homeprice_eu', 'awayprice_eu']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing Decimal odds using American odds
    for price in ['homeprice', 'awayprice']:
        us_col = f'{price}_us'
        eu_col = f'{price}_eu'

        # Where EU odds are missing but US odds are present
        mask = df[eu_col].isna() & df[us_col].notna()
        df.loc[mask, eu_col] = df.loc[mask, us_col].apply(american_to_decimal)
        
        # Where US odds are missing but EU odds are present
        mask = df[us_col].isna() & df[eu_col].notna()
        df.loc[mask, us_col] = df.loc[mask, eu_col].apply(decimal_to_american)
    
    # Fill missing point spreads by copying from the other format if necessary
    df['point_spread_us'] = df['point_spread_us'].fillna(df['point_spread_eu'])
    df['point_spread_eu'] = df['point_spread_eu'].fillna(df['point_spread_us'])

    # Fix inconsistent point spread signs based on moneyline odds
    df = fix_inconsistent_pointspread(df)
    
    # Make sure all odds are valid and enforce valid American odds
    df['homeprice_us'] = df['homeprice_us'].apply(enforce_valid_odds)
    df['awayprice_us'] = df['awayprice_us'].apply(enforce_valid_odds)
    
    return df

def main():
    # Path to the raw data CSV
    csv_path = 'xyzballodds.csv'
    
    # Load and clean data
    df = load_and_clean_data(csv_path)
    print(f"Data loaded and cleaned. {df.shape[0]} records remaining.")
    
    cleaned_csv_path = 'xyzballodds_cleaned.csv'
    df.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned data saved to {cleaned_csv_path}.")

if __name__ == "__main__":
    main()
