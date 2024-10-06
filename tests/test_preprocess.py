# tests/test_preprocess.py

import pytest
import pandas as pd
from src.data.preprocess import handle_time_features, select_columns

def test_handle_time_features():
    data = {'time': ['12:30:45', '08:15:30']}
    df_train = pd.DataFrame(data)
    df_test = pd.DataFrame(data)
    
    df_train, df_test = handle_time_features(df_train, df_test)
    
    assert 'hour' in df_train.columns
    assert 'minute' in df_train.columns
    assert 'time' not in df_train.columns
    assert 'hour' in df_test.columns
    assert 'minute' in df_test.columns
    assert 'time' not in df_test.columns

def test_select_columns():
    data = {
        'bg+1:00': [1, 2],
        'feature1': [0.1, 0.2],
        'activity_type': ['A', 'B']
    }
    df = pd.DataFrame(data)
    numerical_cols, categorical_cols = select_columns(df)
    
    assert 'feature1' in numerical_cols
    assert 'bg+1:00' not in numerical_cols
    assert 'activity_type' in categorical_cols
