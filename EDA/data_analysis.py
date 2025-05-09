import pandas as pd
import numpy as np
from typing import Dict, Tuple

def calculate_basic_stats(df: pd.DataFrame) -> Dict:
    """Calculate basic statistics for temperature data"""
    return {
        'TAVG': df['TAVG'].describe(),
        'TMAX': df['TMAX'].describe(),
        'TMIN': df['TMIN'].describe()
    }

def analyze_seasonal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze seasonal temperature patterns"""
    return df.groupby('Season')['TAVG'].agg(['mean', 'std', 'min', 'max'])

def analyze_yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze yearly temperature trends"""
    return df.groupby('Year')['TAVG'].agg(['mean', 'std', 'min', 'max'])