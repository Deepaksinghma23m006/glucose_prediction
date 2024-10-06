# src/data.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Preprocess categorical columns
    categorical_cols = [col for col in train.columns if 'activity' in col]
    for col in categorical_cols:
        train[col].fillna('None', inplace=True)
        test[col].fillna('None', inplace=True)
        
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
    
    # Preprocess numerical columns
    numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    scaler = MinMaxScaler()
    train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])
    
    return train, test