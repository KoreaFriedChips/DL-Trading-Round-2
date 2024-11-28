import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# Load and preprocess the data
def load_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert data types
    numeric_columns = ['point_spread_us', 'point_spread_eu', 'league_id',
                       'homeprice_us', 'awayprice_us',
                       'homeprice_eu', 'awayprice_eu']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Train LightGBM models for home and away moneyline odds (US format)
def train_models(df, save_dir='models'):
    # Features and targets
    features = ['point_spread_us', 'league_id']
    
    # Model for homeprice_us
    X_home = df[features]
    y_home = df['homeprice_us']
    
    # Model for awayprice_us
    X_away = df[features]
    y_away = df['awayprice_us']
    
    # Split the data
    X_train_home, X_val_home, y_train_home, y_val_home = train_test_split(
        X_home, y_home, test_size=0.2, random_state=42
    )
    
    X_train_away, X_val_away, y_train_away, y_val_away = train_test_split(
        X_away, y_away, test_size=0.2, random_state=42
    )
    
    # Create LightGBM datasets
    lgb_train_home = lgb.Dataset(X_train_home, y_train_home)
    lgb_val_home = lgb.Dataset(X_val_home, y_val_home, reference=lgb_train_home)
    
    lgb_train_away = lgb.Dataset(X_train_away, y_train_away)
    lgb_val_away = lgb.Dataset(X_val_away, y_val_away, reference=lgb_train_away)
    
    # Define parameters
    # Key changes: lowering the number of leaves from 144 to 20, lowering the learning rate from 0.1 to 0.05
    # and adding regularization terms lambda_l1 and lambda_l2 (L1 and L2 regularization reduces overfitting)
    params = {
        'objective': 'regression',
        'metric': ['mae', 'mse'],
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42,
        'num_leaves': 20,       
        'learning_rate': 0.05,       
        'feature_fraction': 0.8,   
        'bagging_fraction': 0.8,   
        'bagging_freq': 5,         
        'lambda_l1': 0.1,          
        'lambda_l2': 0.1           
    }

    
    # Train model for homeprice_us with early stopping to prevent overfitting
    model_home = lgb.train(
        params,
        lgb_train_home,
        valid_sets=[lgb_train_home, lgb_val_home],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    
    # Train model for awayprice_us with early stopping to prevent overfitting
    model_away = lgb.train(
        params,
        lgb_train_away,
        valid_sets=[lgb_train_away, lgb_val_away],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    
    # Print error metrics to evaluate models
    preds_home = model_home.predict(X_val_home, num_iteration=model_home.best_iteration)
    mae_home = mean_absolute_error(y_val_home, preds_home)
    mse_home = mean_squared_error(y_val_home, preds_home)
    map_home = mean_absolute_percentage_error(y_val_home, preds_home)
    cv_scores = cross_val_score(lgb.LGBMRegressor(**params), X_home, y_home, cv=5, scoring='neg_mean_absolute_error')
    print(f"Home Price US Model MAE: {mae_home:.2f}")
    print(f"Home Price US Model MSE: {mse_home:.2f}")
    print(f"Home Price US Model MAPE: {map_home:.2f}%")
    print("Cross-Validation MAE (Home Price):", -np.mean(cv_scores))
    
    preds_away = model_away.predict(X_val_away, num_iteration=model_away.best_iteration)
    mae_away = mean_absolute_error(y_val_away, preds_away)
    mse_away = mean_squared_error(y_val_away, preds_away)
    map_away = mean_absolute_percentage_error(y_val_away, preds_away)
    cv_scores = cross_val_score(lgb.LGBMRegressor(**params), X_away, y_away, cv=5, scoring='neg_mean_absolute_error')
    print(f"Away Price US Model MAE: {mae_away:.2f}")
    print(f"Away Price US Model MSE: {mse_away:.2f}")
    print(f"Away Price US Model MAPE: {map_away:.2f}%")
    print("Cross-Validation MAE (Away Price):", -np.mean(cv_scores))
    
    # Save the models in the save directory
    os.makedirs(save_dir, exist_ok=True)
    home_model_path = os.path.join(save_dir, 'model_home.txt')
    away_model_path = os.path.join(save_dir, 'model_away.txt')
    model_home.save_model(home_model_path)
    model_away.save_model(away_model_path)

    return model_home, model_away

# Define the prediction function
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

# Function to plot the relationships and predictions
def plot_relationships(df, model_home, model_away, samples, league_id=1):
    sns.set(style="whitegrid")
    
    # Filter data for the specified league_id
    df_plot = df[df['league_id'] == league_id]
    
    if df_plot.empty:
        print(f"No data available for league_id={league_id}. Skipping plots for this league.")
        return
    
    # Define the range of point_spread for predictions
    point_spread_min = df_plot['point_spread_us'].min()
    point_spread_max = df_plot['point_spread_us'].max()
    point_spread_range = np.linspace(point_spread_min, point_spread_max, 100)
    
    # Create a DataFrame for predictions
    prediction_df = pd.DataFrame({
        'point_spread_us': point_spread_range,
        'league_id': league_id
    })
    
    pred_home_us = model_home.predict(prediction_df)
    pred_away_us = model_away.predict(prediction_df)
    
    pred_home_us = [enforce_valid_odds(odds) for odds in pred_home_us]
    pred_away_us = [enforce_valid_odds(odds) for odds in pred_away_us]
    
    prediction_df['pred_home_us'] = pred_home_us
    prediction_df['pred_away_us'] = pred_away_us
    
    prediction_df['pred_home_eu'] = prediction_df['pred_home_us'].apply(american_to_decimal)
    prediction_df['pred_away_eu'] = prediction_df['pred_away_us'].apply(american_to_decimal)
    
    # Plot US Point Spread vs Home Moneyline (US)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='point_spread_us', y='homeprice_us', data=df_plot, label='Actual Home Moneyline (US)', color='blue')
    sns.lineplot(x='point_spread_us', y='pred_home_us', data=prediction_df, label='Predicted Home Moneyline (US)', color='red')
    
    # Plot sample predictions
    plotted = False
    for sample in samples:
        if sample['league_id'] == league_id:
            odds = predict_odds(sample['point_spread'], sample['league_id'], model_home, model_away)
            plt.scatter(sample['point_spread'], odds['homeprice_us'], color='green', marker='X', s=100, label='Sample Prediction' if not plotted else "")
            plotted = True
    
    plt.title(f'US Point Spread vs Home Moneyline (US) for League {league_id}')
    plt.xlabel('Point Spread (US)')
    plt.ylabel('Home Moneyline (US)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot US Point Spread vs Away Moneyline (US)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='point_spread_us', y='awayprice_us', data=df_plot, label='Actual Away Moneyline (US)', color='purple')
    sns.lineplot(x='point_spread_us', y='pred_away_us', data=prediction_df, label='Predicted Away Moneyline (US)', color='orange')
    
    # Plot sample predictions
    plotted = False
    for sample in samples:
        if sample['league_id'] == league_id:
            odds = predict_odds(sample['point_spread'], sample['league_id'], model_home, model_away)
            plt.scatter(sample['point_spread'], odds['awayprice_us'], color='green', marker='X', s=100, label='Sample Prediction' if not plotted else "")
            plotted = True
    
    plt.title(f'US Point Spread vs Away Moneyline (US) for League {league_id}')
    plt.xlabel('Point Spread (US)')
    plt.ylabel('Away Moneyline (US)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Path to the cleaned CSV data
    csv_path = 'xyzballodds_cleaned.csv'
    
    # Load the data
    df = load_data(csv_path)
    print(f"Data loaded and cleaned. {df.shape[0]} records remaining.")
    
    extreme_threshold = 200
    extreme_points1 = df[(df['homeprice_us'] > extreme_threshold) | (df['homeprice_us'] < -extreme_threshold)]
    extreme_points2 = df[(df['awayprice_us'] > extreme_threshold) | (df['awayprice_us'] < -extreme_threshold)]
    extreme_points3 = df[(df['point_spread_us'] > extreme_threshold) | (df['point_spread_us'] < -extreme_threshold)]
    df_augmented = pd.concat([df, extreme_points1, extreme_points2, extreme_points3])
    df_augmented = pd.concat([df_augmented, extreme_points1, extreme_points2, extreme_points3])

    # Train models
    model_home, model_away = train_models(df_augmented)
    
    # Print some examples to make sure the predictions make sense
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
    
    # Plotting the relationships and predictions for each league
    print("\nGenerating Plots...")
    unique_leagues = df['league_id'].unique()
    for league in unique_leagues:
        plot_relationships(df_augmented, model_home, model_away, samples, league_id=league)

if __name__ == "__main__":
    main()
   
