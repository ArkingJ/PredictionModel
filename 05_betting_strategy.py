import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

# API Configuration
API_FOOTBALL_KEY = "3ef3b021a60101225b5691fbff2a6679"
BASE_URL = "https://v3.football.api-sports.io"

# Headers for API requests
headers = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': API_FOOTBALL_KEY
}

def load_trained_model():
    """
    Load the trained model from pickle file
    """
    try:
        with open('trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"Loaded {model_data['model_type']} model")
        return model_data
    except FileNotFoundError:
        print("Error: trained_model.pkl not found. Please run model training first.")
        return None

def get_odds_data(fixture_id):
    """
    Fetch odds data for a specific fixture
    Note: This is a placeholder as the free API tier may not provide historical odds
    """
    url = f"{BASE_URL}/odds"
    params = {'fixture': fixture_id}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return data['response']
        else:
            print(f"Error fetching odds for fixture {fixture_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching odds for fixture {fixture_id}: {e}")
        return None

def simulate_odds_data(df):
    """
    Simulate realistic odds data since historical odds may not be available
    This creates synthetic odds based on team performance and market expectations
    """
    print("Simulating odds data (since historical odds may not be available via API)...")
    
    # Create synthetic odds based on team performance
    odds_data = []
    
    for idx, row in df.iterrows():
        # Base probabilities based on team form
        home_form = row.get('home_team_form_last_5', 1.5)  # Default to 1.5 if missing
        away_form = row.get('away_team_form_last_5', 1.5)
        
        # Normalize form to probabilities (0-1 scale)
        home_form_norm = max(0.1, min(0.9, (home_form + 1) / 4))  # Form ranges from 0-3
        away_form_norm = max(0.1, min(0.9, (away_form + 1) / 4))
        
        # Home advantage factor
        home_advantage = 0.1
        
        # Calculate base probabilities
        home_prob = home_form_norm + home_advantage
        away_prob = away_form_norm
        draw_prob = 1 - home_prob - away_prob
        
        # Ensure probabilities sum to 1 and are positive
        total_prob = home_prob + away_prob + draw_prob
        home_prob /= total_prob
        away_prob /= total_prob
        draw_prob /= total_prob
        
        # Add some randomness to make it realistic
        noise = np.random.normal(0, 0.05, 3)
        home_prob = max(0.1, min(0.8, home_prob + noise[0]))
        away_prob = max(0.1, min(0.8, away_prob + noise[1]))
        draw_prob = max(0.05, min(0.4, draw_prob + noise[2]))
        
        # Normalize again
        total_prob = home_prob + away_prob + draw_prob
        home_prob /= total_prob
        away_prob /= total_prob
        draw_prob /= total_prob
        
        # Convert to decimal odds (with bookmaker margin)
        margin = 1.05  # 5% bookmaker margin
        home_odds = margin / home_prob
        away_odds = margin / away_prob
        draw_odds = margin / draw_prob
        
        odds_data.append({
            'fixture_id': row['fixture_id'],
            'home_odds': round(home_odds, 2),
            'away_odds': round(away_odds, 2),
            'draw_odds': round(draw_odds, 2),
            'home_prob_implied': round(home_prob, 4),
            'away_prob_implied': round(away_prob, 4),
            'draw_prob_implied': round(draw_prob, 4)
        })
    
    return pd.DataFrame(odds_data)

def apply_value_betting_strategy(df, odds_df, model_data):
    """
    Apply value betting strategy based on model predictions vs implied odds
    """
    print("Applying value betting strategy...")
    
    # Load model and make predictions
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    
    # Prepare features for prediction
    X = df[feature_columns]
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        predictions = model.predict_proba(X)
    else:
        # For models without predict_proba, use predict
        predictions = model.predict(X)
        # Convert to one-hot encoding
        from sklearn.preprocessing import label_binarize
        predictions = label_binarize(predictions, classes=[0, 1, 2])
    
    # Create results dataframe
    results = []
    
    for idx, row in df.iterrows():
        fixture_id = row['fixture_id']
        odds_row = odds_df[odds_df['fixture_id'] == fixture_id]
        
        if len(odds_row) == 0:
            continue
            
        odds_row = odds_row.iloc[0]
        
        # Get model predictions for this match
        if len(predictions.shape) > 1:
            home_pred = predictions[idx][0]  # Assuming order: Home, Draw, Away
            draw_pred = predictions[idx][1]
            away_pred = predictions[idx][2]
        else:
            # Handle single prediction case
            pred = predictions[idx]
            home_pred = 1.0 if pred == 0 else 0.0
            draw_pred = 1.0 if pred == 1 else 0.0
            away_pred = 1.0 if pred == 2 else 0.0
        
        # Calculate value for each outcome
        home_value = home_pred - odds_row['home_prob_implied']
        draw_value = draw_pred - odds_row['draw_prob_implied']
        away_value = away_pred - odds_row['away_prob_implied']
        
        # Find best value bet (threshold: 0.05 or 5%)
        threshold = 0.05
        best_value = max(home_value, draw_value, away_value)
        best_outcome = None
        
        if home_value >= threshold and home_value == best_value:
            best_outcome = 'H'
            bet_odds = odds_row['home_odds']
            bet_prob = home_pred
            implied_prob = odds_row['home_prob_implied']
        elif draw_value >= threshold and draw_value == best_value:
            best_outcome = 'D'
            bet_odds = odds_row['draw_odds']
            bet_prob = draw_pred
            implied_prob = odds_row['draw_prob_implied']
        elif away_value >= threshold and away_value == best_value:
            best_outcome = 'A'
            bet_odds = odds_row['away_odds']
            bet_prob = away_pred
            implied_prob = odds_row['away_prob_implied']
        
        # Record bet decision
        bet_placed = best_outcome is not None
        bet_amount = 1.0 if bet_placed else 0.0  # Flat staking
        
        # Calculate profit/loss
        if bet_placed:
            if best_outcome == row['result']:
                profit = (bet_odds - 1) * bet_amount
            else:
                profit = -bet_amount
        else:
            profit = 0
        
        results.append({
            'fixture_id': fixture_id,
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'actual_result': row['result'],
            'home_pred': home_pred,
            'draw_pred': draw_pred,
            'away_pred': away_pred,
            'home_odds': odds_row['home_odds'],
            'draw_odds': odds_row['draw_odds'],
            'away_odds': odds_row['away_odds'],
            'home_value': home_value,
            'draw_value': draw_value,
            'away_value': away_value,
            'best_value': best_value,
            'bet_placed': bet_placed,
            'bet_outcome': best_outcome,
            'bet_odds': bet_odds if bet_placed else None,
            'bet_prob': bet_prob if bet_placed else None,
            'implied_prob': implied_prob if bet_placed else None,
            'bet_amount': bet_amount,
            'profit': profit
        })
    
    return pd.DataFrame(results)

def calculate_roi(results_df):
    """
    Calculate Return on Investment and other betting metrics
    """
    print("\n" + "="*60)
    print("BETTING STRATEGY RESULTS")
    print("="*60)
    
    total_bets = results_df['bet_placed'].sum()
    winning_bets = results_df[(results_df['bet_placed']) & (results_df['profit'] > 0)].shape[0]
    total_stake = results_df['bet_amount'].sum()
    total_profit = results_df['profit'].sum()
    
    if total_stake > 0:
        roi = (total_profit / total_stake) * 100
        win_rate = (winning_bets / total_bets) * 100 if total_bets > 0 else 0
    else:
        roi = 0
        win_rate = 0
    
    print(f"Total bets placed: {total_bets}")
    print(f"Winning bets: {winning_bets}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total stake: £{total_stake:.2f}")
    print(f"Total profit/loss: £{total_profit:.2f}")
    print(f"ROI: {roi:.2f}%")
    
    # Value distribution
    print(f"\nValue distribution:")
    value_ranges = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, float('inf'))]
    for low, high in value_ranges:
        if high == float('inf'):
            count = results_df[(results_df['bet_placed']) & (results_df['best_value'] >= low)].shape[0]
            print(f"  Value {low}+: {count} bets")
        else:
            count = results_df[(results_df['bet_placed']) & (results_df['best_value'] >= low) & (results_df['best_value'] < high)].shape[0]
            print(f"  Value {low}-{high}: {count} bets")
    
    return roi, win_rate

def main():
    """
    Main function to run betting strategy simulation
    """
    print("FOOTBALL MATCH PREDICTION - BETTING STRATEGY SIMULATION")
    print("="*70)
    
    # Load trained model
    model_data = load_trained_model()
    if model_data is None:
        return
    
    # Load test data (last season)
    df = pd.read_csv('model_training_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Get test season (last season)
    seasons = sorted(df['season'].unique())
    test_season = seasons[-1]
    test_df = df[df['season'] == test_season].copy()
    
    print(f"Testing betting strategy on season {test_season}")
    print(f"Test matches: {len(test_df)}")
    
    # Simulate odds data
    odds_df = simulate_odds_data(test_df)
    
    # Apply value betting strategy
    results_df = apply_value_betting_strategy(test_df, odds_df, model_data)
    
    # Calculate ROI and performance metrics
    roi, win_rate = calculate_roi(results_df)
    
    # Save results
    results_df.to_csv('betting_results.csv', index=False)
    print(f"\nBetting results saved to 'betting_results.csv'")
    
    # Display sample results
    print(f"\nSample betting decisions:")
    sample_cols = ['home_team', 'away_team', 'actual_result', 'bet_outcome', 'best_value', 'profit']
    print(results_df[results_df['bet_placed']][sample_cols].head(10))
    
    print("\n" + "="*70)
    print("BETTING STRATEGY SIMULATION COMPLETED!")
    print("="*70)
    
    return results_df, roi, win_rate

if __name__ == "__main__":
    results_df, roi, win_rate = main()


