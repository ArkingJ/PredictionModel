import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, classification_report
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load the feature-rich dataset and prepare it for machine learning
    """
    print("Loading model training data...")
    df = pd.read_csv('model_training_data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Separate features from target
    feature_columns = [col for col in df.columns if col not in [
        'fixture_id', 'date', 'home_team', 'away_team', 'goals_home', 
        'goals_away', 'status', 'season', 'round', 'result'
    ]]
    
    X = df[feature_columns]
    y = df['result']
    
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Target distribution: {y.value_counts()}")
    
    return X, y, feature_columns

def encode_target(y):
    """
    Encode the categorical target variable
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Target encoding:")
    for i, label in enumerate(le.classes_):
        print(f"  {label} -> {i}")
    
    return y_encoded, le

def split_data_by_season(X, y, df):
    """
    Split data by season to prevent data leakage
    Use earlier seasons for training, later for testing
    """
    # Get unique seasons and sort them
    seasons = sorted(df['season'].unique())
    print(f"Available seasons: {seasons}")
    
    # Use first 4 seasons for training, last season for testing
    train_seasons = seasons[:-1]
    test_season = seasons[-1]
    
    print(f"Training seasons: {train_seasons}")
    print(f"Test season: {test_season}")
    
    # Create masks
    train_mask = df['season'].isin(train_seasons)
    test_mask = df['season'] == test_season
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Logistic Regression model
    """
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*50)
    
    # Initialize and train the model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = lr_model.predict_proba(X_test)
    y_pred = lr_model.predict(X_test)
    
    # Calculate log loss
    lr_log_loss = log_loss(y_test, y_pred_proba)
    print(f"Logistic Regression Log Loss: {lr_log_loss:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return lr_model, y_pred_proba, lr_log_loss

def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train and evaluate an XGBoost model
    """
    print("\n" + "="*50)
    print("TRAINING XGBOOST MODEL")
    print("="*50)
    
    # Initialize and train the model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = xgb_model.predict_proba(X_test)
    y_pred = xgb_model.predict(X_test)
    
    # Calculate log loss
    xgb_log_loss = log_loss(y_test, y_pred_proba)
    print(f"XGBoost Log Loss: {xgb_log_loss:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return xgb_model, y_pred_proba, xgb_log_loss

def compare_models(lr_log_loss, xgb_log_loss):
    """
    Compare the performance of both models
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    print(f"Logistic Regression Log Loss: {lr_log_loss:.4f}")
    print(f"XGBoost Log Loss: {xgb_log_loss:.4f}")
    
    if lr_log_loss < xgb_log_loss:
        print("\nLogistic Regression performs better (lower log loss)")
        return "logistic_regression"
    else:
        print("\nXGBoost performs better (lower log loss)")
        return "xgboost"

def save_model(model, model_type, feature_columns, label_encoder):
    """
    Save the best performing model
    """
    model_data = {
        'model': model,
        'model_type': model_type,
        'feature_columns': feature_columns,
        'label_encoder': label_encoder
    }
    
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nBest model ({model_type}) saved to 'trained_model.pkl'")

def main():
    """
    Main function to train and evaluate models
    """
    print("FOOTBALL MATCH PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    X, y, feature_columns = load_and_prepare_data()
    
    # Encode target variable
    y_encoded, label_encoder = encode_target(y)
    
    # Split data by season
    df = pd.read_csv('model_training_data.csv')
    X_train, X_test, y_train, y_test = split_data_by_season(X, y_encoded, df)
    
    # Train Logistic Regression
    lr_model, lr_pred_proba, lr_log_loss = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    
    # Train XGBoost
    xgb_model, xgb_pred_proba, xgb_log_loss = train_xgboost(
        X_train, X_test, y_train, y_test
    )
    
    # Compare models
    best_model_type = compare_models(lr_log_loss, xgb_log_loss)
    
    # Save the best model
    if best_model_type == "logistic_regression":
        best_model = lr_model
    else:
        best_model = xgb_model
    
    save_model(best_model, best_model_type, feature_columns, label_encoder)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return best_model, best_model_type

if __name__ == "__main__":
    best_model, best_model_type = main()


