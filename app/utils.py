import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import streamlit as st
from datetime import datetime


def load_models(model_path: str, preprocessor_path: str) -> Tuple:
    """Load the trained model and preprocessor"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

def prepare_input_data(
    tmax_mean: float,
    tmin_mean: float,
    temp_range: float,
    year: int,
    model_type: str
) -> pd.DataFrame:
    """Prepare input data for prediction based on model type"""
    # Data for Ridge model with all 5 features (base case)
    if model_type == 'Ridge Regression':
        day_of_year = datetime.now().timetuple().tm_yday
        month = datetime.now().month
        
        base_data = {
            'TMIN_7day_mean': [tmin_mean],    # Order matches the Ridge model training
            'TMAX_7day_mean': [tmax_mean],
            'Temp_Range': [temp_range],
            'month_sin': [np.sin(2 * np.pi * month / 12)],
            'day_yr_cos': [np.cos(2 * np.pi * day_of_year / 365)]
        }
    # Base data for Random Forest
    else:
        base_data = {
            'TMAX_7day_mean': [tmax_mean],
            'TMIN_7day_mean': [tmin_mean],
            'Temp_Range': [temp_range],
            'Year': [year]
        }
    
    return pd.DataFrame(base_data)

def make_prediction(
    model,
    preprocessor,
    input_data: pd.DataFrame
) -> float:
    """Make temperature prediction"""
    X_processed = preprocessor.transform(input_data)
    return model.predict(X_processed)[0]

def create_temperature_plot(
    tmin: float,
    tmax: float,
    prediction: float
) -> Tuple:
    """Create an improved temperature range visualization with multiple plots"""
    fig = plt.figure(figsize=(15, 8))
    
    # Create a 2x2 subplot layout
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Main prediction plot with confidence interval
    ax1 = fig.add_subplot(gs[0, :])
    x = np.linspace(tmin-2, tmax+2, 100)
    mean_temp = (tmax + tmin) / 2
    std_temp = (tmax - tmin) / 4
    y = np.exp(-((x - mean_temp)**2)/(2*std_temp**2))
    
    # Plot components with confidence interval
    confidence_range = std_temp * 1.96  # 95% confidence interval
    ax1.fill_between(x, y, alpha=0.2, color='lightblue', label='Temperature Range')
    ax1.axvline(prediction, color='red', linestyle='--', linewidth=2, label='Predicted Avg')
    ax1.fill_between([prediction-confidence_range, prediction+confidence_range], 
                    [0, 0], [1.2, 1.2], 
                    color='red', alpha=0.1, 
                    label='95% Confidence Interval')
    ax1.axvline(tmin, color='blue', linestyle=':', linewidth=1, label='Min Temp')
    ax1.axvline(tmax, color='orange', linestyle=':', linewidth=1, label='Max Temp')
    
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Temperature Prediction with Confidence Interval")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar chart comparing temperatures
    ax2 = fig.add_subplot(gs[1, 0])
    temps = ['Min', 'Predicted', 'Max']
    values = [tmin, prediction, tmax]
    colors = ['blue', 'red', 'orange']
    bars = ax2.bar(temps, values, color=colors, alpha=0.6)
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title("Temperature Comparison")
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}°C',
                ha='center', va='bottom')
    
    # 3. Temperature range gauge
    ax3 = fig.add_subplot(gs[1, 1])
    gauge_range = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(gauge_range), np.sin(gauge_range), 'k-', alpha=0.3)
    
    # Convert temperature to angle (mapping temp range to 0-180 degrees)
    temp_range = tmax - tmin
    angle = np.pi * (prediction - tmin) / temp_range
    ax3.plot([0, np.cos(angle)], [0, np.sin(angle)], 'r-', linewidth=2)
    
    ax3.text(0.5, -0.2, f'Current: {prediction:.1f}°C', 
             ha='center', transform=ax3.transAxes)
    ax3.text(-1.1, 0, f'{tmin:.1f}°C', ha='right')
    ax3.text(1.1, 0, f'{tmax:.1f}°C', ha='left')
    
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title("Temperature Gauge")
    
    plt.tight_layout()
    return fig, ax1  # Return main axis for compatibility