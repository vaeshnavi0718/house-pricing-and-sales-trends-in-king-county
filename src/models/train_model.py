import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Tuple, Dict, Any

def split_and_scale_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split data into train/test sets and scale features.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed
        
    Returns:
        Tuple: Scaled training and testing data
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, '../../models/scaler.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_models(X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train multiple regression models.
    
    Args:
        X_train (np.ndarray): Scaled training features
        y_train (pd.Series): Training target
        
    Returns:
        Dict[str, Any]: Dictionary of trained models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Save the model
        joblib.dump(model, f'../../models/{name.lower().replace(" ", "_")}.joblib')
    
    return trained_models

def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate models using various metrics.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models
        X_test (np.ndarray): Scaled test features
        y_test (pd.Series): Test target
        
    Returns:
        pd.DataFrame: Evaluation metrics for each model
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'R2': r2
        })
    
    return pd.DataFrame(results)

def tune_random_forest(X_train: np.ndarray, y_train: pd.Series) -> RandomForestRegressor:
    """
    Perform hyperparameter tuning for Random Forest model.
    
    Args:
        X_train (np.ndarray): Scaled training features
        y_train (pd.Series): Training target
        
    Returns:
        RandomForestRegressor: Best model
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Save the best model
    joblib.dump(grid_search.best_estimator_, '../../models/tuned_random_forest.joblib')
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    # Example usage
    from src.data.process_data import load_data, clean_data, engineer_features, prepare_model_data
    
    # Load and prepare data
    df = load_data("../../data/kc_house_data.csv")
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    X, y = prepare_model_data(df_feat)
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X, y)
    
    # Train models
    models = train_models(X_train_scaled, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test_scaled, y_test)
    print("\nModel Evaluation Results:")
    print(results)
    
    # Tune Random Forest
    best_rf = tune_random_forest(X_train_scaled, y_train)
    print("\nBest Random Forest Parameters:")
    print(best_rf.get_params()) 