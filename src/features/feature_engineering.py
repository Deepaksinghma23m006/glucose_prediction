import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_time(train, test):
    # Convert time to datetime
    train['time'] = pd.to_datetime(train['time'], format='%H:%M:%S')
    test['time'] = pd.to_datetime(test['time'], format='%H:%M:%S')

    # Extract hour and minute
    train['hour'] = train['time'].dt.hour
    train['minute'] = train['time'].dt.minute
    test['hour'] = test['time'].dt.hour
    test['minute'] = test['time'].dt.minute

    # Drop original time column
    train.drop('time', axis=1, inplace=True)
    test.drop('time', axis=1, inplace=True)

    return train, test

def handle_missing_values(train, test, categorical_cols, target):
    for col in categorical_cols:
        train[col] = train[col].fillna('None')
        test[col] = test[col].fillna('None')

    # Handle target missing values if any (optional)
    if target in train.columns:
        train[target] = train[target].fillna(train[target].mean())
    return train, test

def encode_and_scale(train, test, numerical_cols, categorical_cols):
    # Label Encoding for categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        combined_data = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined_data)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # Scaling numerical features
    scaler = MinMaxScaler()
    train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])

    return train, test, scaler

def feature_engineering(train, test, target='bg+1:00'):
    train, test = preprocess_time(train, test)

    numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_cols.remove(target)
    categorical_cols = [col for col in train.columns if 'activity' in col]

    train, test = handle_missing_values(train, test, categorical_cols, target)

    train, test, scaler = encode_and_scale(train, test, numerical_cols, categorical_cols)

    return train, test, numerical_cols, categorical_cols, scaler

if __name__ == "__main__":
    # Example usage
    from data.data_import import load_data
    from features.feature_engineering import feature_engineering

    train, test, submission_df = load_data()
    train, test, numerical_cols, categorical_cols, scaler = feature_engineering(train, test)
    # Save or proceed to model training
