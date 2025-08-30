import requests
import pandas as pd
import time
from datetime import datetime
import json

# API Configuration
API_FOOTBALL_KEY = "3ef3b021a60101225b5691fbff2a6679"
BASE_URL = "https://v3.football.api-sports.io"

# Headers for API requests
headers = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': API_FOOTBALL_KEY
}

def get_league_id(league_name="Premier League", country="England"):
    """Get the league ID for the specified league"""
    url = f"{BASE_URL}/leagues"
    params = {
        'name': league_name,
        'country': country
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['response']:
            return data['response'][0]['league']['id']
    return None

def get_seasons():
    """Get available seasons for the Premier League"""
    league_id = get_league_id()
    if not league_id:
        return []
    
    url = f"{BASE_URL}/leagues"
    params = {'id': league_id}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['response']:
            return [season['year'] for season in data['response'][0]['seasons']]
    return []

def get_fixtures(league_id, season):
    """Get all fixtures for a specific league and season"""
    url = f"{BASE_URL}/fixtures"
    params = {
        'league': league_id,
        'season': season
    }
    
    all_fixtures = []
    page = 1
    
    while True:
        params['page'] = page
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching fixtures for season {season}, page {page}")
            break
            
        data = response.json()
        fixtures = data['response']
        
        if not fixtures:
            break
            
        all_fixtures.extend(fixtures)
        page += 1
        
        # Rate limiting - API allows 100 requests per day for free tier
        time.sleep(1)
    
    return all_fixtures

def get_match_statistics(fixture_id):
    """Get detailed statistics for a specific match"""
    url = f"{BASE_URL}/fixtures/statistics"
    params = {'fixture': fixture_id}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['response']
    return None

def process_fixtures(fixtures):
    """Process fixtures and extract relevant data"""
    processed_data = []
    
    for fixture in fixtures:
        fixture_data = fixture['fixture']
        teams = fixture['teams']
        goals = fixture['goals']
        league = fixture['league']
        
        # Basic match data
        match_info = {
            'fixture_id': fixture_data['id'],
            'date': fixture_data['date'],
            'home_team': teams['home']['name'],
            'away_team': teams['away']['name'],
            'goals_home': goals['home'] if goals['home'] is not None else 0,
            'goals_away': goals['away'] if goals['away'] is not None else 0,
            'status': fixture_data['status']['short'],
            'season': league['season'],
            'round': fixture['league']['round'] if 'round' in fixture['league'] else 'Regular Season'
        }
        
        # Only include finished matches
        if match_info['status'] == 'FT':
            processed_data.append(match_info)
    
    return processed_data

def main():
    """Main function to acquire historical data"""
    print("Starting data acquisition for Premier League...")
    
    # Get league ID
    league_id = get_league_id()
    if not league_id:
        print("Could not find Premier League ID")
        return
    
    print(f"Found Premier League ID: {league_id}")
    
    # Get available seasons (last 5 seasons)
    all_seasons = get_seasons()
    if not all_seasons:
        print("Could not fetch seasons")
        return
    
    # Sort seasons and take last 5
    all_seasons.sort(reverse=True)
    target_seasons = all_seasons[:5]
    print(f"Target seasons: {target_seasons}")
    
    all_matches = []
    
    for season in target_seasons:
        print(f"Fetching fixtures for season {season}...")
        
        fixtures = get_fixtures(league_id, season)
        if fixtures:
            processed_fixtures = process_fixtures(fixtures)
            all_matches.extend(processed_fixtures)
            print(f"Found {len(processed_fixtures)} finished matches for season {season}")
        
        # Rate limiting between seasons
        time.sleep(2)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_matches)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Total matches collected: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save to CSV
    df.to_csv('historical_matches.csv', index=False)
    print("Data saved to historical_matches.csv")
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")

if __name__ == "__main__":
    main()


