# src/pipeline.py
from src.data import preprocess_data
from src.model import train_model
from sklearn.model_selection import train_test_split

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from src
from src.data import preprocess_data
from src.model import train_model

def run_pipeline():
    train, test = preprocess_data("data/raw/train.csv", "data/raw/test.csv")
    
    X = train.drop(['id', 'p_num', 'bg+1:00'], axis=1)
    y = train['bg+1:00']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model, rmse = train_model(X_train, y_train, X_val, y_val)
    print(f"Model trained with RMSE: {rmse}")

if __name__ == "__main__":
    run_pipeline()