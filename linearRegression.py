import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

def american_to_decimal(american_odds):
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

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

def load_data(csv_path):
    
    df = pd.read_csv(csv_path)
    
    # Replace string 'NA' with actual NaN values
    df.replace(['NA', 'na', 'NaN', 999], np.nan, inplace=True)
    
    # Convert data types
    numeric_columns = ['point_spread_us', 'point_spread_eu', 'league_id',
                       'homeprice_us', 'awayprice_us',
                       'homeprice_eu', 'awayprice_eu']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def train_linear_model(df):
    features = ['point_spread_us', 'league_id']
    X = df[features]
    y_home = df['homeprice_us']
    y_away = df['awayprice_us']
    
    # Split the data
    X_train_home, X_val_home, y_train_home, y_val_home = train_test_split(
        X, y_home, test_size=0.2, random_state=42
    )
    
    X_train_away, X_val_away, y_train_away, y_val_away = train_test_split(
        X, y_away, test_size=0.2, random_state=42
    )
    
    # Initialize and train models
    lr_home = LinearRegression()
    lr_home.fit(X_train_home, y_train_home)
    
    lr_away = LinearRegression()
    lr_away.fit(X_train_away, y_train_away)
    
    # Predict
    preds_home = lr_home.predict(X_val_home)
    preds_away = lr_away.predict(X_val_away)
    
    # Evaluate
    mae_home = mean_absolute_error(y_val_home, preds_home)
    mse_home = mean_squared_error(y_val_home, preds_home)
    mape_home = mean_absolute_percentage_error(y_val_home, preds_home)
    
    mae_away = mean_absolute_error(y_val_away, preds_away)
    mse_away = mean_squared_error(y_val_away, preds_away)
    mape_away = mean_absolute_percentage_error(y_val_away, preds_away)
    
    print(f"Linear Regression Home Price US Model MAE: {mae_home:.2f}")
    print(f"Linear Regression Home Price US Model MSE: {mse_home:.2f}")
    print(f"Linear Regression Home Price US Model MAPE: {mape_home:.2f}")
    
    print(f"Away Price US Model MAE: {mae_away:.2f}")
    print(f"Away Price US Model MSE: {mse_away:.2f}")
    print(f"Away Price US Model MAPE: {mape_away:.2f}")
    
    return lr_home, lr_away

def predict_odds(point_spread, league_id, model_home, model_away):
    
    # Create a DataFrame for the input
    input_df = pd.DataFrame({
        'point_spread_us': [point_spread],
        'league_id': [league_id]
    })
    
    # Predict home and away odds
    predicted_home_us = model_home.predict(input_df)[0]
    predicted_away_us = model_away.predict(input_df)[0]
    
    # Enforce valid American odds
    predicted_home_us = enforce_valid_odds(predicted_home_us)
    predicted_away_us = enforce_valid_odds(predicted_away_us)
    
    # Convert to Decimal odds
    predicted_home_eu = american_to_decimal(predicted_home_us)
    predicted_away_eu = american_to_decimal(predicted_away_us)
    
    return {
        'homeprice_us': round(predicted_home_us, 2),
        'awayprice_us': round(predicted_away_us, 2),
        'homeprice_eu': round(predicted_home_eu, 2),
        'awayprice_eu': round(predicted_away_eu, 2)
    }

def main():
    csv_path = 'xyzballodds_cleaned.csv'
    
    # Load data
    df = load_data(csv_path)
    print(f"Data loaded and cleaned. {df.shape[0]} records remaining.")
    extreme_threshold = 200
    extreme_points1 = df[(df['homeprice_us'] > extreme_threshold) | (df['homeprice_us'] < -extreme_threshold)]
    extreme_points2 = df[(df['awayprice_us'] > extreme_threshold) | (df['awayprice_us'] < -extreme_threshold)]
    extreme_points3 = df[(df['point_spread_us'] > extreme_threshold) | (df['point_spread_us'] < -extreme_threshold)]
    df_augmented = pd.concat([df, extreme_points1, extreme_points2, extreme_points3])
    
    # Train models
    model_home, model_away = train_linear_model(df_augmented)
    
    print("\nExample Usage:")
    samples = [
        {'point_spread': 99.5, 'league_id': 1},
        {'point_spread': 0.5, 'league_id': 2},
        {'point_spread': -99.5, 'league_id': 3},
        {'point_spread': 477.5, 'league_id': 1}
    ]
    
    for sample in samples:
        point_spread = sample['point_spread']
        league_id = sample['league_id']
        odds = predict_odds(point_spread, league_id, model_home, model_away)
        print("\nPredicted Odds:")
        print(f"Home Price (US): {odds['homeprice_us']}")
        print(f"Away Price (US): {odds['awayprice_us']}")
        print(f"Home Price (EU): {odds['homeprice_eu']}")
        print(f"Away Price (EU): {odds['awayprice_eu']}")

if __name__ == "__main__":
    main()
    