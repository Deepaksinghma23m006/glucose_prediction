import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils.helpers import print_progress

def train_xgboost(train_df, numerical_cols, categorical_cols, target='bg+1:00'):
    X = train_df.drop(['id', 'p_num', target], axis=1)
    y = train_df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'learning_rate': 0.1,
        'max_depth': 6,
        'device': 'cuda',
        'verbosity': 0,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    num_rounds = 100

    with mlflow.start_run():
        mlflow.log_params(params)
        model = xgb.train(params, dtrain, num_rounds, evals=[(dval, 'eval')], verbose_eval=False)
        y_pred = model.predict(dval)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mlflow.log_metric('rmse', rmse)
        mlflow.xgboost.log_model(model, "model")
        print(f'Root Mean Squared Error: {rmse}')

    return model

if __name__ == "__main__":
    from data.data_import import load_data
    from features.feature_engineering import feature_engineering

    train, test, submission_df = load_data()
    train, test, numerical_cols, categorical_cols, scaler = feature_engineering(train, test)
    model = train_xgboost(train, numerical_cols, categorical_cols)
