import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple

def plot_temperature_distribution(df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot temperature distributions"""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.histplot(df['TAVG'], ax=ax[0], kde=True)
    ax[0].set_title('Average Temperature Distribution')
    
    sns.histplot(df['TMAX'], ax=ax[1], kde=True)
    ax[1].set_title('Maximum Temperature Distribution')
    
    sns.histplot(df['TMIN'], ax=ax[2], kde=True)
    ax[2].set_title('Minimum Temperature Distribution')
    
    plt.tight_layout()
    return fig, ax

def plot_yearly_trend(df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot yearly temperature trends"""
    yearly_avg = df.groupby('Year')['TAVG'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_avg.plot(ax=ax)
    ax.set_title('Yearly Temperature Trend')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Temperature (°C)')
    
    return fig, ax