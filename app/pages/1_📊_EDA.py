import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from EDA.data_analysis import calculate_basic_stats, analyze_seasonal_patterns, analyze_yearly_trends

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')

# Set base font size for better readability on smaller plots
plt.rcParams.update({'font.size': 10})

st.title("Exploratory Data Analysis")

st.markdown("""
This page presents the exploratory data analysis of Tanzania's temperature data. Understanding historical temperature patterns 
is crucial for climate analysis and helps contextualize predictions made by our models.
""")

# Load data
try:
    data_path = os.path.join(project_root, 'data', 'tanzania_weather_features_complete.csv') # data path
    df = pd.read_csv(data_path)  
    
    # Basic Statistics
    st.header("Basic Temperature Statistics")
    st.markdown("""
    These statistics provide a numerical summary of Tanzania's temperature patterns:
    - **Mean**: The average temperature across all recorded data
    - **Std**: Standard deviation, showing how much temperatures typically vary from the mean
    - **Min/Max**: The extreme temperature values recorded
    - **Percentiles (25%, 50%, 75%)**: Show the distribution of temperatures, with 50% being the median
    
    A high standard deviation indicates significant temperature variability, which is important for climate resilience planning.
    """)
    
    stats = calculate_basic_stats(df)

    # Convert dictionary to DataFrame for better display
    stats_df = pd.DataFrame({
        'Statistic': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
        'Average Temperature (°C)': stats['TAVG'],
        'Maximum Temperature (°C)': stats['TMAX'],
        'Minimum Temperature (°C)': stats['TMIN']
    })

    # Display as a styled table
    st.dataframe(
        stats_df.set_index('Statistic'),
        use_container_width=True,
        hide_index=False
    )

    # Temperature Distributions with Matplotlib
    st.header("Temperature Distributions")
    st.markdown("""
    These histograms show how temperatures are distributed across the dataset:
    - **Bell-shaped distributions** indicate normal temperature patterns
    - **Skewed distributions** may suggest climate anomalies or seasonal biases in the data
    - **Multiple peaks** could indicate distinct climate regimes or seasonal patterns
    
    In the context of climate change, shifts in these distributions over time can indicate changing temperature patterns.
    """)
    
    # Create Matplotlib histogram for temperature distributions
    fig, axes = plt.subplots(1, 3, figsize=(7, 4))
    
    # Add histograms for each temperature type
    axes[0].hist(df['TAVG'], bins=25, alpha=0.7, color='#49a0b5')
    axes[0].set_title("Average Temperature", fontsize=11)
    axes[0].set_xlabel("Temperature (°C)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    
    axes[1].hist(df['TMAX'], bins=25, alpha=0.7, color='#ff6d00')
    axes[1].set_title("Maximum Temperature", fontsize=11)
    axes[1].set_xlabel("Temperature (°C)", fontsize=10)
    
    axes[2].hist(df['TMIN'], bins=25, alpha=0.7, color='#32ab60')
    axes[2].set_title("Minimum Temperature", fontsize=11)
    axes[2].set_xlabel("Temperature (°C)", fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Seasonal Patterns
    st.header("Seasonal Temperature Patterns")
    st.markdown("""
    This table shows how temperatures vary across different seasons in Tanzania:
    - **Mean**: Average temperature for each season
    - **Std**: How much temperatures vary within each season
    - **Min/Max**: The extreme temperatures recorded in each season
    
    Understanding seasonal patterns is crucial for agriculture planning, tourism, and energy demand forecasting.
    Unusual seasonal patterns may indicate climate change impacts.
    """)
    seasonal_stats = analyze_seasonal_patterns(df)
    st.dataframe(seasonal_stats, use_container_width=True)
    
    # Seasonal Temperature Visualization with Matplotlib
    st.subheader("Seasonal Temperature Visualization")
    
    st.markdown("""
    This visualization shows Tanzania's seasonal temperature patterns with variability:
    
    - **Bar height**: Represents the average temperature for each season
    - **Error bars**: Show the standard deviation, indicating how much temperatures typically vary within each season
    - **Temperature range**: The minimum to maximum temperatures recorded in each season
    
    **What this means in the climate context:**
    
    - **Wet Season (Mar-May)**: Higher temperatures with moderate variability, critical for crop germination and early growth
    - **Dry Season (Jun-Oct)**: Cooler temperatures with less variability, important for harvest and post-harvest activities
    - **Short Rains (Nov-Dec)**: Rising temperatures with increased variability, affecting planting decisions for short-season crops
    - **Hot Season (Jan-Feb)**: Peak temperatures with high variability, creating heat stress for crops and livestock
    
    Climate change may alter these patterns by:
    - Increasing variability (longer error bars)
    - Shifting seasonal timing (earlier or later onset)
    - Raising average temperatures across all seasons
    - Creating more extreme temperature events
    
    These changes impact agricultural planning, water resource management, and human health considerations.
    """)
    
    # Create Matplotlib bar chart for seasonal patterns
    seasons = seasonal_stats.index
    means = seasonal_stats['mean']
    std = seasonal_stats['std']
    min_temps = seasonal_stats['min']
    max_temps = seasonal_stats['max']
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Define colors
    colors = ['#49a0b5', '#32ab60', '#ff6d00', '#ffb600']
    
    # Add bars
    bars = ax.bar(
        seasons,
        means,
        yerr=std,
        capsize=8,
        color=colors,
        alpha=0.7
    )
    
    # Add range indicators as annotations
    for i, (season, mean, std_val, min_temp, max_temp) in enumerate(zip(seasons, means, std, min_temps, max_temps)):
        ax.annotate(
            f"Range: {min_temp:.1f}-{max_temp:.1f}°C",
            xy=(i, mean + std_val + 0.5),
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    # Update layout
    ax.set_title("Seasonal Temperature Patterns with Variability", fontsize=12)
    ax.set_xlabel("Season", fontsize=10)
    ax.set_ylabel("Temperature (°C)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Yearly Trends with Matplotlib
    st.header("Yearly Temperature Trends")
    st.markdown("""
    This graph shows how average temperatures have changed over the years:
    - **Upward trends** may indicate warming due to climate change
    - **Downward trends** could suggest cooling patterns
    - **Flat periods** show temperature stability
    - **Variability** (ups and downs) may indicate climate oscillations like El Niño/La Niña

    **Important context for interpretation:**
    - This dataset covers approximately 30 years (1990-2022), which may be too short to identify long-term climate trends
    - Regional factors like the Indian Ocean Dipole can create multi-decadal cooling or warming periods
    - The slight cooling trend observed may reflect regional climate patterns and dataset limitations rather than contradicting global warming
    
    Long-term temperature trends are key indicators of climate change. Tanzania, like many regions, 
    has experienced gradual warming over recent decades, which impacts agriculture, water resources, 
    and biodiversity.
    """)
    
    # Enhanced yearly trend visualization
    yearly_stats = analyze_yearly_trends(df)
    yearly_avg = yearly_stats['mean']
    
    # Use actual years for trend calculation
    years = np.array(yearly_stats.index.astype(int))
    z = np.polyfit(years, yearly_avg, 1)
    p = np.poly1d(z)
    trend_years = np.array(range(min(years), max(years)+1))

    # Calculate confidence intervals for the trend line
    n = len(years)
    mean_x = np.mean(years)
    std_err = np.sqrt(np.sum((yearly_avg - p(years))**2) / (n-2) / np.sum((years - mean_x)**2))
    conf_interval = std_err * 1.96  # 95% confidence interval

    # Create upper and lower confidence bands
    upper_bound = p(trend_years) + conf_interval
    lower_bound = p(trend_years) - conf_interval

    # Calculate 5-year moving average
    window_size = 5
    if len(yearly_avg) >= window_size:
        # Create pandas Series for rolling calculation
        temp_series = pd.Series(yearly_avg.values, index=yearly_stats.index)
        rolling_avg = temp_series.rolling(window=window_size, center=True).mean()
    
    # Create Matplotlib line chart for yearly trends
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Add min-max range
    ax.fill_between(
        yearly_stats.index,
        yearly_stats['min'],
        yearly_stats['max'],
        color='#1f77b4',
        alpha=0.2,
        label='Min-Max Range'
    )
    
    # Add yearly average line
    ax.plot(
        yearly_stats.index,
        yearly_stats['mean'],
        'o-',
        color='#1f77b4',
        linewidth=1.5,
        markersize=6,
        label='Yearly Average'
    )
    
    # Add trend line
    ax.plot(
        trend_years,
        p(trend_years),
        '--',
        color='red',
        linewidth=1.5,
        label=f'Trend: {z[0]:.4f}°C/year'
    )
    
    # Update layout
    ax.set_title('Yearly Temperature Trend (1990-2022)', fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create custom legend with more descriptive labels
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.2, label='Temperature Range (Min to Max)'),
        Line2D([0], [0], color='#1f77b4', marker='o', linestyle='-', markersize=6, linewidth=1.5, label='Yearly Average Temp'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=f'Trend: {z[0]:.4f}°C/year')
    ]
    
    # Position legend outside the plot area at the bottom
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.35), fontsize=9, ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom margin to make room for the legend
    
    st.pyplot(fig)

    
    # Temperature Anomalies with Matplotlib
    st.subheader("Temperature Anomalies")
    st.markdown("""
    Temperature anomalies show deviations from the long-term average, highlighting unusual warming or cooling periods:
    - **Positive anomalies (red)** indicate warmer than average periods
    - **Negative anomalies (blue)** indicate cooler than average periods
    
    Increasing frequency and intensity of positive anomalies is a key indicator of climate change.
    """)
    
    # Calculate the baseline average (e.g., first 10 years or a reference period)
    baseline_years = sorted(df['Year'].unique())[:10]  # First 10 years as baseline
    baseline_avg = df[df['Year'].isin(baseline_years)]['TAVG'].mean()
    
    # Calculate yearly anomalies
    yearly_anomalies = yearly_avg - baseline_avg
    
    # Create Matplotlib bar chart for anomalies
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Create colors based on anomaly values
    colors = ['#ff4136' if x > 0 else '#1f77b4' for x in yearly_anomalies]
    
    # Create bars
    bars = ax.bar(
        yearly_anomalies.index,
        yearly_anomalies,
        color=colors,
        alpha=0.7
    )
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # Update layout
    ax.set_title(f'Temperature Anomalies (Baseline: {baseline_years[0]}-{baseline_years[-1]} Average)', fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Temperature Anomaly (°C)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#ff4136', alpha=0.7, label='Warmer than baseline'),
        Patch(facecolor='#1f77b4', alpha=0.7, label='Cooler than baseline')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add climate impact section
    st.header("Climate Implications")
    st.markdown("""
    ### What These Patterns Mean for Tanzania
    
    The temperature patterns shown above have several implications for Tanzania:
    
    1. **Agriculture**: Temperature trends affect growing seasons, crop yields, and pest prevalence
    
    2. **Water Resources**: Higher temperatures increase evaporation rates, affecting water availability
    
    3. **Health**: Temperature extremes can impact public health through heat stress and disease vectors
    
    4. **Biodiversity**: Changing temperatures affect ecosystems and wildlife habitats
    
    5. **Energy**: Temperature patterns influence energy demand for cooling and heating
    
    Understanding these patterns helps in developing climate adaptation strategies and resilience planning.
    """)
    
except Exception as e:
    st.error("Error loading data. Please ensure data files are present.")
    st.exception(e)