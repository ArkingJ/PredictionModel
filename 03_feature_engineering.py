import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_team_features(df):
    """
    Create team-specific features using rolling averages
    Only uses information known before a match
    """
    # Sort by date to ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Initialize feature columns
    feature_columns = []
    
    # For each team, calculate rolling averages
    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    
    for team in all_teams:
        print(f"Processing features for team: {team}")
        
        # Get all matches for this team (both home and away)
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        team_matches = team_matches.sort_values('date').reset_index(drop=True)
        
        # Initialize rolling average columns
        team_matches['team_goals_scored_last_5'] = np.nan
        team_matches['team_goals_conceded_last_5'] = np.nan
        team_matches['team_form_last_5'] = np.nan
        team_matches['team_avg_goals_scored_last_5'] = np.nan
        team_matches['team_avg_goals_conceded_last_5'] = np.nan
        
        # Calculate rolling features
        for i in range(len(team_matches)):
            if i < 5:  # Need at least 5 previous matches
                continue
                
            # Get last 5 matches
            last_5_matches = team_matches.iloc[i-5:i]
            
            goals_scored = []
            goals_conceded = []
            points = []
            
            for _, match in last_5_matches.iterrows():
                if match['home_team'] == team:
                    # Team was home
                    goals_scored.append(match['goals_home'])
                    goals_conceded.append(match['goals_away'])
                    
                    # Calculate points
                    if match['goals_home'] > match['goals_away']:
                        points.append(3)  # Win
                    elif match['goals_home'] == match['goals_away']:
                        points.append(1)  # Draw
                    else:
                        points.append(0)  # Loss
                else:
                    # Team was away
                    goals_scored.append(match['goals_away'])
                    goals_conceded.append(match['goals_home'])
                    
                    # Calculate points
                    if match['goals_away'] > match['goals_home']:
                        points.append(3)  # Win
                    elif match['goals_away'] == match['goals_home']:
                        points.append(1)  # Draw
                    else:
                        points.append(0)  # Loss
            
            # Calculate averages
            team_matches.loc[team_matches.index[i], 'team_goals_scored_last_5'] = np.mean(goals_scored)
            team_matches.loc[team_matches.index[i], 'team_goals_conceded_last_5'] = np.mean(goals_conceded)
            team_matches.loc[team_matches.index[i], 'team_form_last_5'] = np.mean(points)
            team_matches.loc[team_matches.index[i], 'team_avg_goals_scored_last_5'] = np.mean(goals_scored)
            team_matches.loc[team_matches.index[i], 'team_avg_goals_conceded_last_5'] = np.mean(goals_conceded)
        
        # Add these features back to the main dataframe
        for col in ['team_goals_scored_last_5', 'team_goals_conceded_last_5', 
                   'team_form_last_5', 'team_avg_goals_scored_last_5', 'team_avg_goals_conceded_last_5']:
            feature_columns.append(col)
            
            # Map back to main dataframe
            for idx, row in team_matches.iterrows():
                main_idx = df[df['fixture_id'] == row['fixture_id']].index
                if len(main_idx) > 0:
                    df.loc[main_idx[0], col] = row[col]
    
    return df, feature_columns

def create_home_away_features(df):
    """
    Create home and away specific features
    """
    # Home team features
    df['home_team_avg_goals_scored_last_5'] = np.nan
    df['home_team_avg_goals_conceded_last_5'] = np.nan
    df['home_team_form_last_5'] = np.nan
    
    # Away team features
    df['away_team_avg_goals_scored_last_5'] = np.nan
    df['away_team_avg_goals_conceded_last_5'] = np.nan
    df['away_team_form_last_5'] = np.nan
    
    # Map team features to home/away
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Home team features
        if pd.notna(row['team_avg_goals_scored_last_5']):
            df.loc[idx, 'home_team_avg_goals_scored_last_5'] = row['team_avg_goals_scored_last_5']
        if pd.notna(row['team_avg_goals_conceded_last_5']):
            df.loc[idx, 'home_team_avg_goals_conceded_last_5'] = row['team_avg_goals_conceded_last_5']
        if pd.notna(row['team_form_last_5']):
            df.loc[idx, 'home_team_form_last_5'] = row['team_form_last_5']
        
        # Away team features (need to find away team's data)
        away_team_data = df[(df['home_team'] == away_team) | (df['away_team'] == away_team)]
        if len(away_team_data) > 0:
            # Find the most recent match for away team before this match
            away_team_data = away_team_data[away_team_data['date'] < row['date']].sort_values('date')
            if len(away_team_data) > 0:
                latest_away_match = away_team_data.iloc[-1]
                if pd.notna(latest_away_match['team_avg_goals_scored_last_5']):
                    df.loc[idx, 'away_team_avg_goals_scored_last_5'] = latest_away_match['team_avg_goals_scored_last_5']
                if pd.notna(latest_away_match['team_avg_goals_conceded_last_5']):
                    df.loc[idx, 'away_team_avg_goals_conceded_last_5'] = latest_away_match['team_avg_goals_conceded_last_5']
                if pd.notna(latest_away_match['team_form_last_5']):
                    df.loc[idx, 'away_team_form_last_5'] = latest_away_match['team_form_last_5']
    
    return df

def create_head_to_head_features(df):
    """
    Create head-to-head features
    """
    df['head_to_head_goals_home'] = np.nan
    df['head_to_head_goals_away'] = np.nan
    df['head_to_head_matches'] = np.nan
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        match_date = row['date']
        
        # Find previous matches between these teams
        h2h_matches = df[
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ]
        
        # Only consider matches before current match
        h2h_matches = h2h_matches[h2h_matches['date'] < match_date].sort_values('date')
        
        if len(h2h_matches) > 0:
            # Take last 5 matches
            recent_h2h = h2h_matches.tail(5)
            
            home_goals = []
            away_goals = []
            
            for _, h2h_match in recent_h2h.iterrows():
                if h2h_match['home_team'] == home_team:
                    # Current home team was home in this match
                    home_goals.append(h2h_match['goals_home'])
                    away_goals.append(h2h_match['goals_away'])
                else:
                    # Current home team was away in this match
                    home_goals.append(h2h_match['goals_away'])
                    away_goals.append(h2h_match['goals_home'])
            
            if home_goals:
                df.loc[idx, 'head_to_head_goals_home'] = np.mean(home_goals)
                df.loc[idx, 'head_to_head_goals_away'] = np.mean(away_goals)
                df.loc[idx, 'head_to_head_matches'] = len(recent_h2h)
    
    return df

def create_additional_features(df):
    """
    Create additional useful features
    """
    # Goal difference features
    df['home_team_goal_diff_last_5'] = df['home_team_avg_goals_scored_last_5'] - df['home_team_avg_goals_conceded_last_5']
    df['away_team_goal_diff_last_5'] = df['away_team_avg_goals_scored_last_5'] - df['away_team_avg_goals_conceded_last_5']
    
    # Form difference
    df['form_difference'] = df['home_team_form_last_5'] - df['away_team_form_last_5']
    
    # Season features
    df['season'] = df['season'].astype(int)
    
    # Day of week
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Month
    df['month'] = df['date'].dt.month
    
    return df

def main():
    """
    Main function to create all features
    """
    print("Loading cleaned historical data...")
    df = pd.read_csv('cleaned_historical_matches.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Original dataset shape: {df.shape}")
    
    print("\nCreating team features...")
    df, feature_columns = create_team_features(df)
    
    print("\nCreating home/away specific features...")
    df = create_home_away_features(df)
    
    print("\nCreating head-to-head features...")
    df = create_head_to_head_features(df)
    
    print("\nCreating additional features...")
    df = create_additional_features(df)
    
    # Remove rows with NaN values (first few matches of each season)
    print("\nRemoving rows with NaN values...")
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    print(f"Removed {initial_rows - final_rows} rows with NaN values")
    
    # Display feature information
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Features created: {len([col for col in df.columns if col not in ['fixture_id', 'date', 'home_team', 'away_team', 'goals_home', 'goals_away', 'status', 'season', 'round', 'result']])}")
    
    # Show sample of features
    feature_cols = [col for col in df.columns if col not in ['fixture_id', 'date', 'home_team', 'away_team', 'goals_home', 'goals_away', 'status', 'season', 'round', 'result']]
    print(f"\nFeature columns: {feature_cols}")
    
    # Save the feature-rich dataset
    df.to_csv('model_training_data.csv', index=False)
    print("\nFeature-rich dataset saved to 'model_training_data.csv'")
    
    # Display sample data
    print("\nSample of final dataset:")
    display_cols = ['date', 'home_team', 'away_team', 'goals_home', 'goals_away', 'result'] + feature_cols[:5]
    print(df[display_cols].head())
    
    return df

if __name__ == "__main__":
    df = main()


