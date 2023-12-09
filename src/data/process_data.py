import pandas as pd
import numpy as np
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the King County housing dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    df_clean = df_clean.fillna({
        'waterfront': 0,
        'view': 0,
        'yr_renovated': 0
    })
    
    # Remove outliers (houses with more than 10 bedrooms or bathrooms)
    df_clean = df_clean[
        (df_clean['bedrooms'] <= 10) & 
        (df_clean['bathrooms'] <= 10)
    ]
    
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    # Create a copy to avoid modifying the original
    df_feat = df.copy()
    
    # Convert date to datetime
    df_feat['date'] = pd.to_datetime(df_feat['date'])
    
    # Extract year and month
    df_feat['year'] = df_feat['date'].dt.year
    df_feat['month'] = df_feat['date'].dt.month
    
    # Calculate house age
    df_feat['house_age'] = df_feat['year'] - df_feat['yr_built']
    
    # Calculate price per square foot
    df_feat['price_per_sqft'] = df_feat['price'] / df_feat['sqft_living']
    
    # Create renovation flag
    df_feat['is_renovated'] = (df_feat['yr_renovated'] > 0).astype(int)
    
    # Create total rooms feature
    df_feat['total_rooms'] = df_feat['bedrooms'] + df_feat['bathrooms']
    
    return df_feat

def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling by selecting features and target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
    """
    # Select features for modeling
    features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'waterfront', 'view', 'condition', 'grade',
        'house_age', 'price_per_sqft', 'is_renovated', 'total_rooms'
    ]
    
    X = df[features]
    y = df['price']
    
    return X, y

if __name__ == "__main__":
    # Example usage
    df = load_data("../../data/kc_house_data.csv")
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    X, y = prepare_model_data(df_feat)
    
    print("Data shape:", df.shape)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape) 