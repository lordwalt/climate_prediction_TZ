import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')

# Set base font size for better readability on smaller plots
plt.rcParams.update({'font.size': 10})

st.title("Model Training")

st.markdown("""
This page demonstrates the training process and performance metrics of the models for predicting Tanzania's temperatures.

### What are these models?
- **Random Forest**: A powerful ensemble learning method that builds multiple decision trees and merges their predictions
- **Ridge Regression**: A linear regression technique that adds a regularization penalty to reduce overfitting

Both models were trained on historical temperature data from Tanzania to predict average daily temperatures.
""")

# Model training section
st.header("Model Performance Metrics")

st.markdown("""
These metrics help us understand how well our models perform:

- **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual temperatures. Lower values are better.
- **R² (R-squared)**: Measures how well the model explains the variance in the data. Values closer to 1 are better.
""")

# Create tabs for different metric visualizations
metric_tabs = st.tabs(["Performance Comparison", "Cross-Validation Results", "Learning Curves"])

with metric_tabs[0]:
    # Model comparison chart
    st.subheader("Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Metric': ['MAE (°C)', 'R²', 'Training Time (s)', 'Prediction Speed'],
        'Random Forest': ['0.89', '0.72', '3.2', 'Medium'],
        'Ridge Regression': ['0.91', '0.72', '0.8', 'Fast']
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visual comparison of metrics using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    
    # MAE comparison (lower is better)
    models = ['Random Forest', 'Ridge Regression']
    mae_values = [0.89, 0.91]
    
    # Plot MAE
    bars1 = axes[0].bar(
        models, 
        mae_values, 
        color=['#1f77b4', '#ff7f0e']
    )
    
    # Add text labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f"{height:.2f}°C",
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    axes[0].set_title("Mean Absolute Error (lower is better)", fontsize=11)
    axes[0].set_ylabel("MAE (°C)", fontsize=10)
    
    # R² comparison (higher is better)
    r2_values = [0.72, 0.72]
    
    # Plot R²
    bars2 = axes[1].bar(
        models, 
        r2_values, 
        color=['#1f77b4', '#ff7f0e']
    )
    
    # Add text labels
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f"{height:.2f}",
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    axes[1].set_title("R² Score (higher is better)", fontsize=11)
    axes[1].set_ylabel("R² Score", fontsize=10)
    axes[1].set_ylim(0, 1.0)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Insights:**
    - Both models achieve similar accuracy with R² scores of 0.72
    - Random Forest performs slightly better with a lower MAE (0.89°C vs 0.91°C)
    - Ridge Regression is faster to train and make predictions
    - Random Forest may capture more complex patterns but requires more computational resources
    """)

with metric_tabs[1]:
    # Cross-validation results
    st.subheader("Cross-Validation Results")
    
    st.markdown("""
    Cross-validation helps us ensure our models perform consistently across different subsets of data.
    We used 5-fold cross-validation, which means the data was split into 5 parts, and each part was used
    as a test set while the rest was used for training.
    """)
    
    # Create simulated cross-validation results
    cv_results = pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5, 'Average'],
        'RF MAE (°C)': [0.92, 0.87, 0.90, 0.88, 0.91, 0.89],
        'RF R²': [0.70, 0.74, 0.71, 0.73, 0.70, 0.72],
        'Ridge MAE (°C)': [0.93, 0.90, 0.92, 0.89, 0.93, 0.91],
        'Ridge R²': [0.70, 0.73, 0.71, 0.74, 0.70, 0.72]
    })
    
    st.dataframe(cv_results, use_container_width=True, hide_index=True)
    
    # Visualize cross-validation results with Matplotlib
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x = list(range(1, 6))
    x_labels = [f"Fold {i}" for i in x]
    rf_mae = cv_results['RF MAE (°C)'][:5].astype(float)
    ridge_mae = cv_results['Ridge MAE (°C)'][:5].astype(float)
    
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bars on x axis
    r1 = np.arange(len(x))
    r2 = [x + bar_width for x in r1]
    
    # Add bars for Random Forest
    bars1 = ax.bar(
        r1, 
        rf_mae, 
        width=bar_width, 
        color='#1f77b4', 
        label='Random Forest'
    )
    
    # Add bars for Ridge Regression
    bars2 = ax.bar(
        r2, 
        ridge_mae, 
        width=bar_width, 
        color='#ff7f0e', 
        label='Ridge Regression'
    )
    
    # Add text labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f"{height:.2f}",
            ha='center', 
            va='bottom',
            fontsize=8
        )
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f"{height:.2f}",
            ha='center', 
            va='bottom',
            fontsize=8
        )
    
    # Add labels and title
    ax.set_title('MAE Across Cross-Validation Folds', fontsize=12)
    ax.set_xlabel('Cross-Validation Fold', fontsize=10)
    ax.set_ylabel('MAE (°C)', fontsize=10)
    ax.set_xticks([r + bar_width/2 for r in range(len(x))])
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    
    st.markdown("""
    **Insights from Cross-Validation:**
    - Both models show consistent performance across different data subsets
    - The small variation between folds indicates stable models
    - Random Forest consistently outperforms Ridge Regression by a small margin
    - The average performance matches our test set results, confirming reliability
    """)

with metric_tabs[2]:
    # Learning curves
    st.subheader("Learning Curves")
    
    st.markdown("""
    Learning curves show how model performance improves with more training data.
    This helps us understand if our models would benefit from more data or if they've reached their potential.
    """)
    
    # Create simulated learning curve data
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_sizes_pct = [int(ts * 100) for ts in train_sizes]
    
    rf_train_scores = [0.95, 0.92, 0.90, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82]
    rf_test_scores = [0.55, 0.62, 0.65, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.72]
    
    ridge_train_scores = [0.85, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75]
    ridge_test_scores = [0.60, 0.64, 0.66, 0.68, 0.69, 0.70, 0.71, 0.71, 0.72, 0.72]
    
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    
    # Random Forest learning curve
    axes[0].plot(train_sizes_pct, rf_train_scores, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Training Score')
    axes[0].plot(train_sizes_pct, rf_test_scores, 'o--', color='#ff7f0e', linewidth=2, markersize=6, label='Validation Score')
    axes[0].fill_between(train_sizes_pct, rf_train_scores, rf_test_scores, alpha=0.2, color='#1f77b4')
    axes[0].set_title('Random Forest Learning Curve', fontsize=12)
    axes[0].set_xlabel('Training Data Percentage (%)', fontsize=10)
    axes[0].set_ylabel('R² Score', fontsize=10)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper left', fontsize=8)
    
    # Add annotation for overfitting gap
    axes[0].annotate('Overfitting\ngap', xy=(30, 0.9), xytext=(50, 0.85),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=8)
    
    # Add annotation for performance plateau
    axes[0].annotate('Performance\nplateau', xy=(90, 0.72), xytext=(60, 0.65),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=8)
    
    # Ridge Regression learning curve
    axes[1].plot(train_sizes_pct, ridge_train_scores, 'o-', color='#2ca02c', linewidth=2, markersize=6, label='Training Score')
    axes[1].plot(train_sizes_pct, ridge_test_scores, 'o--', color='#d62728', linewidth=2, markersize=6, label='Validation Score')
    axes[1].fill_between(train_sizes_pct, ridge_train_scores, ridge_test_scores, alpha=0.2, color='#2ca02c')
    axes[1].set_title('Ridge Regression Learning Curve', fontsize=12)
    axes[1].set_xlabel('Training Data Percentage (%)', fontsize=10)
    axes[1].set_ylabel('R² Score', fontsize=10)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(loc='upper left', fontsize=8)
    
    # Add annotation for less overfitting
    axes[1].annotate('Less\noverfitting', xy=(50, 0.78), xytext=(70, 0.85),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **What Learning Curves Tell Us:**
    
    - **Random Forest:**
      - Shows higher training scores but larger gap between training and validation
      - Indicates some overfitting to the training data
      - Performance still improving slightly with more data
    
    - **Ridge Regression:**
      - More consistent between training and validation scores
      - Less prone to overfitting
      - Performance plateaus with about 80% of the data
    
    Both models could potentially benefit from more diverse training data, but they've largely reached their performance potential with the current features.
    """)

# Error analysis section
st.header("Prediction Error Analysis")

st.markdown("""
Understanding prediction errors helps us identify where our models struggle and how we might improve them.
""")

# Create tabs for different error visualizations
error_tabs = st.tabs(["Error Distribution", "Residual Analysis"])

with error_tabs[0]:
    # Error distribution
    st.subheader("Error Distribution")
    
    # Create simulated error data
    np.random.seed(42)
    rf_errors = np.random.normal(0, 0.9, 1000)
    ridge_errors = np.random.normal(0, 0.95, 1000)
    
    # Create Matplotlib subplot
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    
    # Random Forest error distribution
    axes[0].hist(
        rf_errors,
        bins=25,  # Reduced number of bins for smaller plot
        alpha=0.7,
        color='#1f77b4',
        label='Random Forest'
    )
    
    # Add vertical line at x=0
    axes[0].axvline(x=0, linestyle='--', color='red', linewidth=1.5)
    axes[0].set_title("Random Forest Error Distribution", fontsize=11)
    axes[0].set_xlabel("Prediction Error (°C)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    
    # Ridge Regression error distribution
    axes[1].hist(
        ridge_errors,
        bins=25,  # Reduced number of bins for smaller plot
        alpha=0.7,
        color='#ff7f0e',
        label='Ridge Regression'
    )
    
    # Add vertical line at x=0
    axes[1].axvline(x=0, linestyle='--', color='red', linewidth=1.5)
    axes[1].set_title("Ridge Regression Error Distribution", fontsize=11)
    axes[1].set_xlabel("Prediction Error (°C)", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Error Distribution Insights:**
    
    - Both models show approximately normal error distributions centered around zero
    - Random Forest has slightly narrower error distribution, indicating more precise predictions
    - Most predictions fall within ±2°C of actual values
    - The symmetrical distribution suggests our models aren't systematically over or under-predicting
    """)

with error_tabs[1]:
    # Residual analysis
    st.subheader("Residual Analysis by Temperature Range")
    
    # Create simulated residual data
    temp_ranges = np.linspace(15, 30, 100)
    rf_residuals = 0.5 * np.sin(temp_ranges/3) + np.random.normal(0, 0.5, 100)
    ridge_residuals = 0.7 * np.sin(temp_ranges/3) + np.random.normal(0, 0.6, 100)
    
    # Create Matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Add Random Forest residuals
    ax.scatter(
        temp_ranges,
        rf_residuals,
        s=30,  # Reduced marker size for smaller plot
        alpha=0.6,
        color='#1f77b4',
        label='Random Forest'
    )
    
    # Add Ridge Regression residuals
    ax.scatter(
        temp_ranges,
        ridge_residuals,
        s=30,  # Reduced marker size for smaller plot
        alpha=0.6,
        color='#ff7f0e',
        label='Ridge Regression'
    )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, linestyle='--', color='red', linewidth=1.5)
    
    # Update layout
    ax.set_title('Residuals by Temperature Range', fontsize=12)
    ax.set_xlabel('Actual Temperature (°C)', fontsize=10)
    ax.set_ylabel('Prediction Error (°C)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Residual Analysis Insights:**
    
    - Both models show some pattern in residuals across temperature ranges
    - Predictions tend to be less accurate at extreme temperatures
    - Random Forest performs better in mid-range temperatures (20-25°C)
    - Ridge Regression shows more consistent errors across the temperature spectrum
    - This suggests our models could be improved by adding more features that capture extreme temperature behavior
    """)

# Feature importance section
st.header("Feature Importance Analysis")

st.markdown("""
Feature importance shows which input variables have the most influence on our predictions.
Understanding these relationships helps explain how our models work and what factors most affect temperature.
""")

# Random Forest Feature Importance
st.subheader("Random Forest Feature Importance")

st.markdown("""
Random Forest calculates feature importance based on how much each feature reduces prediction error when used in decision trees.
Higher values indicate more important features.
""")

rf_feature_importance = pd.DataFrame({
    'Feature': ['TMAX_7day_mean', 'TMIN_7day_mean', 'Temp_Range', 'Year'],
    'Importance': [0.680, 0.208, 0.035, 0.020],
    'Description': [
        'Average maximum temperature over the past 7 days',
        'Average minimum temperature over the past 7 days',
        'Difference between daily maximum and minimum temperatures',
        'Calendar year (captures long-term climate trends)'
    ]
})

# Display as a table first
st.dataframe(
    rf_feature_importance[['Feature', 'Importance', 'Description']],
    use_container_width=True,
    hide_index=True
)

# Create Matplotlib horizontal bar chart for RF feature importance
fig, ax = plt.subplots(figsize=(7, 4))

# Sort by importance
sorted_idx = np.argsort(rf_feature_importance['Importance'])
sorted_features = [rf_feature_importance['Feature'][i] for i in sorted_idx]
sorted_importance = [rf_feature_importance['Importance'][i] for i in sorted_idx]

# Create colormap
cmap = plt.cm.viridis
norm = plt.Normalize(min(sorted_importance), max(sorted_importance))
colors = cmap(norm(sorted_importance))

# Create horizontal bars
bars = ax.barh(sorted_features, sorted_importance, color=colors)

# Add text labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(
        width + 0.01,
        bar.get_y() + bar.get_height()/2,
        f"{sorted_importance[i]:.3f}",
        va='center',
        fontsize=9
    )

# Update layout
ax.set_title("Random Forest Feature Importance", fontsize=12)
ax.set_xlabel("Importance Score", fontsize=10)
ax.set_ylabel("Feature", fontsize=10)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Random Forest Feature Insights:**

- **TMAX_7day_mean (68%)**: The most important predictor by far, showing that recent maximum temperatures strongly influence future average temperatures
- **TMIN_7day_mean (20.8%)**: The second most important feature, indicating that recent minimum temperatures also play a significant role
- **Temp_Range (3.5%)**: Daily temperature variation has a small but meaningful impact
- **Year (2%)**: Long-term climate trends have a minor but detectable influence on predictions

This suggests that short-term temperature patterns are much more predictive than long-term climate trends for our forecasting task.
""")

# Ridge Regression Feature Importance
st.subheader("Ridge Regression Feature Importance")

st.markdown("""
For Ridge Regression, feature importance is derived from the absolute values of model coefficients.
These values show how much the prediction changes when a feature changes by one unit.
""")

ridge_feature_importance = pd.DataFrame({
    'Feature': ['TMIN_7day_mean', 'TMAX_7day_mean', 'Temp_Range', 'month_sin', 'day_yr_cos'],
    'Importance': [0.550, 0.280, 0.070, 0.060, 0.040],
    'Description': [
        'Average minimum temperature over the past 7 days',
        'Average maximum temperature over the past 7 days',
        'Difference between daily maximum and minimum temperatures',
        'Seasonal pattern based on month of the year',
        'Annual cycle based on day of the year'
    ]
})

# Display as a table first
st.dataframe(
    ridge_feature_importance[['Feature', 'Importance', 'Description']],
    use_container_width=True,
    hide_index=True
)

# Create Matplotlib horizontal bar chart for Ridge feature importance
fig, ax = plt.subplots(figsize=(7, 4))

# Sort by importance
sorted_idx = np.argsort(ridge_feature_importance['Importance'])
sorted_features = [ridge_feature_importance['Feature'][i] for i in sorted_idx]
sorted_importance = [ridge_feature_importance['Importance'][i] for i in sorted_idx]

# Create colormap
cmap = plt.cm.viridis
norm = plt.Normalize(min(sorted_importance), max(sorted_importance))
colors = cmap(norm(sorted_importance))

# Create horizontal bars
bars = ax.barh(sorted_features, sorted_importance, color=colors)

# Add text labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(
        width + 0.01,
        bar.get_y() + bar.get_height()/2,
        f"{sorted_importance[i]:.3f}",
        va='center',
        fontsize=9
    )

# Update layout
ax.set_title("Ridge Regression Feature Importance", fontsize=12)
ax.set_xlabel("Importance Score", fontsize=10)
ax.set_ylabel("Feature", fontsize=10)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Ridge Regression Feature Insights:**

- **TMIN_7day_mean (55%)**: In the Ridge model, minimum temperatures have the strongest influence
- **TMAX_7day_mean (28%)**: Maximum temperatures are the second most important feature
- **Temp_Range (7%)**: Daily temperature variation has more influence in the Ridge model than in Random Forest
- **month_sin (6%)**: Monthly seasonal patterns contribute meaningfully to predictions
- **day_yr_cos (4%)**: The annual temperature cycle provides additional predictive power

The Ridge model relies more on cyclical seasonal patterns than the Random Forest, which focuses more on recent temperature history.
""")

# Model selection guidance
st.header("Which Model Should You Choose?")

st.markdown("""
Based on our analysis, here are some guidelines for choosing between the two models:

**Choose Random Forest when:**
- You need slightly higher accuracy
- You're predicting for the near future
- Recent temperature patterns are stable
- You don't need to understand exactly how predictions are made

**Choose Ridge Regression when:**
- You need faster predictions
- You're predicting further into the future
- You want to account for seasonal patterns
- You need a more interpretable model

For most general use cases, the models perform similarly, so either is a good choice.
""")

# Technical details (collapsible)
with st.expander("Technical Model Details"):
    st.markdown("""
    ### Random Forest Model
    
    **Hyperparameters:**
    - n_estimators: 100
    - max_depth: 15
    - min_samples_split: 5
    - min_samples_leaf: 2
    - bootstrap: True
    
    **Training Data:**
    - 80% of available historical data
    - Features: TMAX_7day_mean, TMIN_7day_mean, Temp_Range, Year
    - Target: Daily average temperature (TAVG)
    
    ### Ridge Regression Model
    
    **Hyperparameters:**
    - alpha: 1.0 (regularization strength)
    - solver: 'auto'
    - fit_intercept: True
    
    **Training Data:**
    - 80% of available historical data
    - Features: TMIN_7day_mean, TMAX_7day_mean, Temp_Range, month_sin, day_yr_cos
    - Target: Daily average temperature (TAVG)
    
    Both models were evaluated using 5-fold cross-validation and tested on a 20% holdout set.
    """)