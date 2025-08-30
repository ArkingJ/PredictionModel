import pandas as pd
import numpy as np
import requests
import pickle
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration - Set reference date to a realistic date from current season
REFERENCE_DATE = datetime(2024, 12, 20)  # Friday, December 20, 2024 (current season)

# API Configuration
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY', "3ef3b021a60101225b5691fbff2a6679")
BASE_URL = "https://v3.football.api-sports.io"

# Headers for API requests
headers = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': API_FOOTBALL_KEY
}

def get_league_id(league_name="Premier League", country="England"):
    """Get the league ID for the specified league"""
    url = f"{BASE_URL}/leagues"
    params = {'name': league_name, 'country': country}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['response']:
            return data['response'][0]['league']['id']
    return None

def get_real_fixtures(league_id):
    """Get real fixtures from API-Football for the current season"""
    print("Fetching real fixtures from API-Football...")
    print(f"Reference date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
    
    url = f"{BASE_URL}/fixtures"
    params = {
        'league': league_id,
        'season': 2024  # Current season
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error fetching fixtures: {response.status_code}")
        return [], []
    
    data = response.json()
    fixtures = data['response']
    print(f"Found {len(fixtures)} fixtures for season 2024")
    
    # Separate upcoming and recent fixtures based on reference date
    upcoming_fixtures = []
    recent_fixtures = []
    
    for fixture in fixtures:
        try:
            fixture_date = datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00'))
            
            # Check if fixture is upcoming (after reference date)
            if fixture_date > REFERENCE_DATE:
                upcoming_fixtures.append(fixture)
            # Check if fixture is recent (within 14 days before reference date)
            elif fixture_date <= REFERENCE_DATE and fixture_date >= REFERENCE_DATE - timedelta(days=14):
                recent_fixtures.append(fixture)
        except Exception as e:
            print(f"Error parsing fixture date: {e}")
            continue
    
    print(f"Fixtures after {REFERENCE_DATE.strftime('%Y-%m-%d')}: {len(upcoming_fixtures)}")
    print(f"Fixtures within 14 days before {REFERENCE_DATE.strftime('%Y-%m-%d')}: {len(recent_fixtures)}")
    
    return upcoming_fixtures, recent_fixtures

def create_predictions_from_real_fixtures(upcoming_fixtures, recent_fixtures):
    """Create predictions from real API fixtures"""
    print("Creating predictions from real API fixtures...")
    
    all_predictions = []
    
    # Process upcoming fixtures
    for fixture in upcoming_fixtures:
        fixture_data = fixture['fixture']
        teams = fixture['teams']
        
        prediction = {
            'fixture_id': fixture_data['id'],
            'date': fixture_data['date'],
            'home_team': teams['home']['name'],
            'away_team': teams['away']['name'],
            'predictions': {
                'home_win': 0.4,
                'draw': 0.3,
                'away_win': 0.3
            },
            'predicted_outcome': 'H',
            'confidence': 0.4,
            'match_type': 'upcoming'
        }
        all_predictions.append(prediction)
    
    # Process recent fixtures
    for fixture in recent_fixtures:
        fixture_data = fixture['fixture']
        teams = fixture['teams']
        goals = fixture.get('goals', {})
        
        # Get actual scores if available
        goals_home = goals.get('home', 0) if goals.get('home') is not None else 0
        goals_away = goals.get('away', 0) if goals.get('away') is not None else 0
        
        # Determine actual result
        if goals_home > goals_away:
            actual_result = 'H'
        elif goals_home < goals_away:
            actual_result = 'A'
        else:
            actual_result = 'D'
        
        prediction = {
            'fixture_id': fixture_data['id'],
            'date': fixture_data['date'],
            'home_team': teams['home']['name'],
            'away_team': teams['away']['name'],
            'predictions': {
                'home_win': 0.4,
                'draw': 0.3,
                'away_win': 0.3
            },
            'predicted_outcome': 'H',
            'confidence': 0.4,
            'match_type': 'recent',
            'actual_result': actual_result,
            'goals_home': goals_home,
            'goals_away': goals_away
        }
        all_predictions.append(prediction)
    
    return all_predictions

def save_predictions(predictions):
    """Save predictions to JSON file in docs folder"""
    os.makedirs('docs', exist_ok=True)
    
    output_file = 'docs/predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {output_file}")
    return output_file

def main():
    """Main function to update predictions"""
    print("FOOTBALL MATCH PREDICTION - UPDATE PREDICTIONS")
    print("="*60)
    print(f"Using reference date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
    print("="*60)
    
    # Get Premier League ID
    league_id = get_league_id()
    if not league_id:
        print("Error: Could not find Premier League ID.")
        return []
    
    print(f"Found Premier League ID: {league_id}")
    
    # Get real fixtures from API
    upcoming_fixtures, recent_fixtures = get_real_fixtures(league_id)
    
    if upcoming_fixtures or recent_fixtures:
        print("✓ Found real fixtures in API!")
        predictions = create_predictions_from_real_fixtures(upcoming_fixtures, recent_fixtures)
    else:
        print("✗ No real fixtures found for the reference date.")
        return []
    
    # Save predictions
    output_file = save_predictions(predictions)
    
    # Display summary
    upcoming_count = len([p for p in predictions if p['match_type'] == 'upcoming'])
    recent_count = len([p for p in predictions if p['match_type'] == 'recent'])
    
    print(f"\n" + "="*60)
    print("SUMMARY:")
    print(f"Upcoming fixtures: {upcoming_count}")
    print(f"Recent results: {recent_count}")
    print(f"Total predictions: {len(predictions)}")
    print("="*60)
    
    # Display sample upcoming fixtures
    upcoming = [p for p in predictions if p['match_type'] == 'upcoming'][:5]
    print(f"\nSample upcoming fixtures:")
    for pred in upcoming:
        print(f"{pred['date'][:10]}: {pred['home_team']} vs {pred['away_team']}")
    
    # Display sample recent results
    recent = [p for p in predictions if p['match_type'] == 'recent'][:5]
    print(f"\nSample recent results:")
    for pred in recent:
        print(f"{pred['date'][:10]}: {pred['home_team']} {pred['goals_home']}-{pred['goals_away']} {pred['away_team']} ({pred['actual_result']})")
    
    print(f"\n" + "="*60)
    print("PREDICTIONS UPDATED SUCCESSFULLY!")
    print(f"Output file: {output_file}")
    print("="*60)
    
    return predictions

if __name__ == "__main__":
    predictions = main()
e 