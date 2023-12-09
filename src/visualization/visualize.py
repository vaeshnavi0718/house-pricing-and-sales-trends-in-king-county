import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def set_plot_style():
    """Set the style for all plots."""
    plt.style.use('seaborn')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def plot_price_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot the distribution of house prices.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='price', bins=50)
    plt.title('Distribution of House Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation matrix of numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str, optional): Path to save the plot
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(model, feature_names: List[str], save_path: str = None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): List of feature names
        save_path (str, optional): Path to save the plot
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_price_by_feature(df: pd.DataFrame, feature: str, save_path: str = None):
    """
    Plot house prices by a specific feature.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature to plot against price
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=feature, y='price')
    plt.title(f'House Prices by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_price_trends(df: pd.DataFrame, save_path: str = None):
    """
    Plot price trends over time.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str, optional): Path to save the plot
    """
    # Convert date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Calculate monthly average prices
    monthly_prices = df.groupby(df['date'].dt.to_period('M'))['price'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_prices.plot()
    plt.title('Average House Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Price ($)')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_location_heatmap(df: pd.DataFrame, save_path: str = None):
    """
    Create a heatmap of house prices by location.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(df['long'], df['lat'], c=df['price'], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Price ($)')
    plt.title('House Prices by Location')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Example usage
    from src.data.process_data import load_data, clean_data, engineer_features
    
    # Load and prepare data
    df = load_data("../../data/kc_house_data.csv")
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    
    # Set plot style
    set_plot_style()
    
    # Create visualizations
    plot_price_distribution(df_feat, '../../visualizations/price_distribution.png')
    plot_correlation_matrix(df_feat, '../../visualizations/correlation_matrix.png')
    plot_price_by_feature(df_feat, 'bedrooms', '../../visualizations/price_by_bedrooms.png')
    plot_price_trends(df_feat, '../../visualizations/price_trends.png')
    plot_location_heatmap(df_feat, '../../visualizations/location_heatmap.png') 