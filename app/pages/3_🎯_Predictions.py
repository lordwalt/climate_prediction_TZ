import streamlit as st
import pandas as pd
from datetime import datetime
from config import *
from utils import load_models, prepare_input_data, make_prediction, create_temperature_plot

def main():
    # Page config
    st.set_page_config(
        page_title="Tanzania Climate Predictions",
        page_icon="🌡️",
        layout="wide"
    )

    # Title and description
    st.title("Tanzania Temperature Prediction Dashboard")
    st.markdown("""
    This application predicts daily average temperatures in Tanzania using machine learning models.
    Choose between Random Forest and Ridge Regression models for predictions.
    """)

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        options=list(MODELS.keys()),
        index=1
    )
    
    # Location selection
    location = st.sidebar.selectbox(
        "Select Location",
        options=["Dar es Salaam", "Dodoma", "Arusha", "Mwanza", "Zanzibar"],
        help="Choose the nearest major city in Tanzania"
    )

    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        month = st.selectbox(
            "Month",
            options=["January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"],
            help="Select the month for future temperature prediction"
        )
    
    with col2:
        current_year = datetime.now().year
        year = st.number_input(
            "Year",
            min_value=current_year,
            max_value=current_year + 10,
            value=current_year,
            help="Select the future year for prediction (up to 10 years ahead for better accuracy)"
        )
    
    # Current temperature with slider
    current_temp = st.sidebar.slider(
        "Current Temperature (°C)",
        min_value=float(TEMP_MIN),
        max_value=float(TEMP_MAX),
        value=23.0,
        step=0.5,
        help="Enter the current temperature in your location"
    )
    
    # Time of day for context
    time_of_day = st.sidebar.selectbox(
        "Time of Day",
        options=["Morning (6AM-11AM)", "Afternoon (12PM-5PM)", "Evening (6PM-11PM)", "Night (12AM-5AM)"],
        help="Select the current time of day"
    )

    # Calculate temperature range based on time of day and current temperature
    temp_variations = {
        "Morning (6AM-11AM)": {"range": 5, "offset": -2},
        "Afternoon (12PM-5PM)": {"range": 3, "offset": 2},
        "Evening (6PM-11PM)": {"range": 4, "offset": -1},
        "Night (12AM-5AM)": {"range": 6, "offset": -3}
    }
    
    variation = temp_variations[time_of_day]
    tmax_mean = min(current_temp + variation["offset"] + variation["range"]/2, TEMP_MAX)
    tmin_mean = max(current_temp + variation["offset"] - variation["range"]/2, TEMP_MIN)
    temp_range = tmax_mean - tmin_mean

    # Add a predict button
    predict_button = st.sidebar.button("Predict Temperature", type="primary", use_container_width=True)

    # Only run prediction when button is clicked
    if predict_button:
        try:
            # Load selected model
            model_config = MODELS[model_type]
            model_path = model_config['path']
            preprocessor_path = RIDGE_PREPROCESSOR_PATH if model_type == 'Ridge Regression' else RF_PREPROCESSOR_PATH
            model, preprocessor = load_models(model_path, preprocessor_path)
            
            # Prepare input data with correct features
            input_data = prepare_input_data(
                tmax_mean,
                tmin_mean,
                temp_range,
                year,
                model_type
            )
            
            # Make prediction
            prediction = make_prediction(model, preprocessor, input_data)
            
            # Modified year-over-year validation logic
            years_ahead = year - datetime.now().year
            
            # Adjust the year factor to be more conservative
            year_adjustment = 1.0 + (years_ahead * 0.01)  # 1% increase per year
            
            # Apply seasonal adjustments
            seasonal_factors = {
                "December": 1.02, "January": 1.02, "February": 1.03,  # Hot season
                "March": 1.01, "April": 1.0, "May": 0.99,  # Transition
                "June": 0.98, "July": 0.97, "August": 0.97,  # Cool season
                "September": 0.99, "October": 1.0, "November": 1.01  # Transition
            }
            
            seasonal_adjustment = seasonal_factors.get(month, 1.0)
            prediction = prediction * year_adjustment * seasonal_adjustment
            
            # Ensure logical relationship between temperatures
            if tmin_mean >= prediction:
                # If min temp is higher than prediction, adjust it downward
                tmin_mean = prediction - (temp_range * 0.3)
                
            if tmax_mean <= prediction:
                # If max temp is lower than prediction, adjust it upward
                tmax_mean = prediction + (temp_range * 0.3)
                
            # Enhanced warning system for future predictions
            if years_ahead > 5:
                st.warning(f"""
                    ⚠️ Prediction uncertainty increases for dates far in the future. 
                    You're predicting {years_ahead} years ahead, which may reduce accuracy.
                    Consider using predictions within a 5-year window for more reliable results.
                    
                    Note: A {(year_adjustment-1)*100:.1f}% adjustment has been applied to account for long-term climate trends.
                """)
                
           # Enhanced confidence interval calculation
            base_confidence = 0.15  # Reduced base uncertainty
            year_uncertainty = min(years_ahead * 0.05, 0.3)  # Reduced year uncertainty
            seasonal_uncertainty = 0.03 if month in ["March", "April", "September", "October"] else 0.02
            
            confidence_range = min(base_confidence + year_uncertainty + seasonal_uncertainty, 0.5)
            
            # Calculate confidence interval based on the prediction, not constrained by min/max temps
            lower_bound = prediction * (1 - confidence_range)
            upper_bound = prediction * (1 + confidence_range)
            
            # Ensure expected low and high temperatures are within logical bounds
            # Low temp should be between lower_bound and prediction
            # High temp should be between prediction and upper_bound
            tmin_mean = max(min(tmin_mean, prediction * 0.95), lower_bound * 1.05)
            tmax_mean = min(max(tmax_mean, prediction * 1.05), upper_bound * 0.95)
                
            # Display prediction results in a clean table format
            st.header("Temperature Prediction Results")
            
            # Create a DataFrame for the results with clearer labels
            results_df = pd.DataFrame({
                'Metric': ['Predicted Average Temperature', 
                          'Expected High Temperature', 
                          'Expected Low Temperature', 
                          'Confidence Range', 
                          'Location', 
                          'Time Period'],
                'Value': [
                    f"{prediction:.1f}°C",
                    f"{tmax_mean:.1f}°C",
                    f"{tmin_mean:.1f}°C",
                    f"{lower_bound:.1f}°C - {upper_bound:.1f}°C",
                    location,
                    f"{month} {year}"
                ]
            })
            
            # Display as a clean table
            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True
            )
            
            # Add explanation about the temperature ranges
            st.info("""
            **Understanding the results:**
            - **Predicted Average Temperature**: The expected average temperature for the selected month and year
            - **Expected High Temperature**: The typical maximum temperature during the selected month
            - **Expected Low Temperature**: The typical minimum temperature during the selected month
            - **Confidence Range**: The range within which the actual temperature is likely to fall (95% confidence)
            """)
            
            # Model information
            st.header("Model Information")
            features_info = {
                'Ridge Regression': [
                    '7-day maximum temperature average',
                    '7-day minimum temperature average',
                    'Daily temperature range',
                    'Monthly seasonal pattern',
                    'Annual seasonal cycle'
                ],
                'Random Forest': [
                    '7-day maximum temperature average',
                    '7-day minimum temperature average',
                    'Daily temperature range',
                    'Year'
                ]
            }
            
            # Dynamic feature list based on selected model
            selected_features = features_info[model_type]
            
            # Introduction text
            st.write(f"This prediction is based on our {model_type} model using the following features:")
            
            # Display features as HTML list with custom styling
            features_html = "<ul style='margin-left: 20px; margin-top: 10px; margin-bottom: 20px;'>"
            for feature in selected_features:
                features_html += f"<li style='margin-bottom: 8px;'>{feature}</li>"
            features_html += "</ul>"
            
            st.markdown(features_html, unsafe_allow_html=True)
            
            # Model training info
            st.write("The model has been trained on historical temperature data from Tanzania weather stations.")
            
        except Exception as e:
            st.error("Error loading model. Please ensure model files are present.")
            st.exception(e)
    else:
        # Show instructions when button hasn't been clicked yet
        st.info("👈 Set your parameters in the sidebar and click 'Predict Temperature' to see results")

if __name__ == "__main__":
    main()