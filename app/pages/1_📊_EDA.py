import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from EDA.data_analysis import calculate_basic_stats, analyze_seasonal_patterns, analyze_yearly_trends

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

    # Temperature Distributions with Plotly
    st.header("Temperature Distributions")
    st.markdown("""
    These histograms show how temperatures are distributed across the dataset:
    - **Bell-shaped distributions** indicate normal temperature patterns
    - **Skewed distributions** may suggest climate anomalies or seasonal biases in the data
    - **Multiple peaks** could indicate distinct climate regimes or seasonal patterns
    
    In the context of climate change, shifts in these distributions over time can indicate changing temperature patterns.
    """)
    
    # Create Plotly histogram for temperature distributions
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=("Average Temperature", "Maximum Temperature", "Minimum Temperature"))
    
    # Add traces for each temperature type
    fig.add_trace(
        go.Histogram(x=df['TAVG'], nbinsx=30, name="Average Temp", 
                     marker_color='rgba(73, 160, 181, 0.7)'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=df['TMAX'], nbinsx=30, name="Maximum Temp", 
                     marker_color='rgba(255, 109, 0, 0.7)'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Histogram(x=df['TMIN'], nbinsx=30, name="Minimum Temp", 
                     marker_color='rgba(50, 171, 96, 0.7)'),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Temperature Distributions in Tanzania",
        bargap=0.05,
        template="plotly_white"
    )
    
    # Update x and y axis titles for each subplot
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=3)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Seasonal Temperature Visualization with Plotly
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
    
    # Create Plotly bar chart for seasonal patterns
    seasons = seasonal_stats.index
    means = seasonal_stats['mean']
    std = seasonal_stats['std']
    min_temps = seasonal_stats['min']
    max_temps = seasonal_stats['max']
    
    # Create custom hover text
    hover_text = [f"Season: {season}<br>" +
                 f"Mean: {mean:.1f}°C<br>" +
                 f"Std Dev: {std:.2f}°C<br>" +
                 f"Range: {min_temp:.1f}-{max_temp:.1f}°C"
                 for season, mean, std, min_temp, max_temp 
                 in zip(seasons, means, std, min_temps, max_temps)]
    
    # Create the bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=seasons,
        y=means,
        error_y=dict(type='data', array=std, visible=True),
        hovertext=hover_text,
        hoverinfo='text',
        marker_color=['rgba(73, 160, 181, 0.7)', 'rgba(50, 171, 96, 0.7)', 
                      'rgba(255, 109, 0, 0.7)', 'rgba(255, 182, 0, 0.7)']
    ))
    
    # Add range indicators as annotations
    for i, season in enumerate(seasons):
        fig.add_annotation(
            x=i,
            y=means[i] + std[i] + 0.5,
            text=f"Range: {min_temps[i]:.1f}-{max_temps[i]:.1f}°C",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title="Seasonal Temperature Patterns with Variability",
        xaxis_title="Season",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Yearly Trends with Plotly
    st.header("Yearly Temperature Trends")
    st.markdown("""
    This graph shows how average temperatures have changed over the years:
    - **Upward trends** may indicate warming due to climate change
    - **Downward trends** could suggest cooling patterns
    - **Flat periods** show temperature stability
    - **Variability** (ups and downs) may indicate climate oscillations like El Niño/La Niña
    
    Long-term temperature trends are key indicators of climate change. Tanzania, like many regions, 
    has experienced gradual warming over recent decades, which impacts agriculture, water resources, 
    and biodiversity.
    """)
    
    # Enhanced yearly trend visualization
    yearly_stats = analyze_yearly_trends(df)
    yearly_avg = yearly_stats['mean']
    
    # Create Plotly line chart for yearly trends
    fig = go.Figure()
    
    # Add yearly average line
    fig.add_trace(go.Scatter(
        x=yearly_stats.index,
        y=yearly_stats['mean'],
        mode='lines+markers',
        name='Yearly Average',
        line=dict(color='rgb(31, 119, 180)', width=2),
        marker=dict(size=6)
    ))
    
    # Use actual years for trend calculation
    years = np.array(yearly_stats.index.astype(int))
    z = np.polyfit(years, yearly_avg, 1)
    p = np.poly1d(z)
    trend_years = np.array(range(min(years), max(years)+1))
    fig.add_trace(go.Scatter(
    x=trend_years,
    y=p(trend_years),
    mode='lines',
    name=f'Trend: {z[0]:.4f}°C/year',
    line=dict(color='red', width=2, dash='dash')
))
    
    # Add min-max range
    fig.add_trace(go.Scatter(
        x=yearly_stats.index,
        y=yearly_stats['max'],
        mode='lines',
        name='Maximum',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_stats.index,
        y=yearly_stats['min'],
        mode='lines',
        name='Min-Max Range',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    # Update layout
    fig.update_layout(
        title='Yearly Temperature Trend with Min-Max Range',
        xaxis_title='Year',
        yaxis_title='Temperature (°C)',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature Anomalies with Plotly
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
    
    # Create Plotly bar chart for anomalies
    colors = ['rgba(255, 65, 54, 0.7)' if x > 0 else 'rgba(31, 119, 180, 0.7)' for x in yearly_anomalies]
    
    fig = go.Figure(go.Bar(
        x=yearly_anomalies.index,
        y=yearly_anomalies,
        marker_color=colors,
        hovertemplate='Year: %{x}<br>Anomaly: %{y:.2f}°C<extra></extra>'
    ))
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=min(yearly_anomalies.index),
        y0=0,
        x1=max(yearly_anomalies.index),
        y1=0,
        line=dict(color="black", width=1.5, dash="solid")
    )
    
    # Update layout
    fig.update_layout(
        title=f'Temperature Anomalies (Baseline: {baseline_years[0]}-{baseline_years[-1]} Average)',
        xaxis_title='Year',
        yaxis_title='Temperature Anomaly (°C)',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
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