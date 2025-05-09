import os

# Model settings
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Model paths
RF_MODEL_PATH = os.path.join(project_root, 'models', 'rf_model.pkl')
RIDGE_MODEL_PATH = os.path.join(project_root, 'models', 'ridge_model.pkl')
RIDGE_PREPROCESSOR_PATH = os.path.join(project_root, 'models', 'ridge_preprocessor.pkl')
RF_PREPROCESSOR_PATH = os.path.join(project_root, 'models', 'rf_preprocessor.pkl')

# Feature settings for different models
RF_FEATURES = [
    'TMAX_7day_mean',    # 0.680 importance
    'TMIN_7day_mean',    # 0.208 importance
    'Temp_Range',        # 0.035 importance
    'Year'               # 0.020 importance
]

RIDGE_FEATURES = [
    'TMIN_7day_mean',    # 1.617791
    'TMAX_7day_mean',    # 0.925014
    'Temp_Range',        # 0.452410
    'month_sin',         # 0.223166
    'day_yr_cos'         # 0.143792
]

# Temperature bounds
TEMP_MIN = 10.0
TEMP_MAX = 35.0
RANGE_MIN = 0.0
RANGE_MAX = 20.0

# Model types with their features
MODELS = {
    'Random Forest': {'path': RF_MODEL_PATH, 'features': RF_FEATURES},
    'Ridge Regression': {'path': RIDGE_MODEL_PATH, 'features': RIDGE_FEATURES}
}