# src/models/train.py

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yaml
import logging

def load_config(config_path='config.yaml'):
    """Load configuration parameters."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_xgboost(train_path, config):
    """Train XGBoost model."""
    train = pd.read_csv(train_path)
    X = train.drop(['id', 'p_num', 'bg+1:00'], axis=1)
    y = train['bg+1:00']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        config['xgboost']['params'],
        dtrain,
        config['xgboost']['num_rounds'],
        evals=[(dval, 'eval')],
        verbose_eval=config['xgboost'].get('verbose_eval', False)
    )
    
    return model, X_val, y_val

def train_lightgbm(train_path, config):
    """Train LightGBM model."""
    train = pd.read_csv(train_path)
    X = train.drop(['id', 'p_num', 'bg_1hour'], axis=1)
    y = train['bg_1hour']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        config['lightgbm']['params'],
        lgb_train,
        valid_sets=[lgb_eval],
        num_boost_round=config['lightgbm']['num_rounds'],
        verbose_eval=config['lightgbm'].get('verbose_eval', False)
    )
    
    return model, X_val, y_val

def train_catboost(train_path, config):
    """Train CatBoost model."""
    train = pd.read_csv(train_path)
    X = train.drop(['id', 'p_num', 'bg+1:00'], axis=1)
    y = train['bg+1:00']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = CatBoostRegressor(**config['catboost']['params'])
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    return model, X_val, y_val

def main():
    config = load_config()
    
    # Train XGBoost
    xgb_model, xgb_X_val, xgb_y_val = train_xgboost(
        config['data']['train_processed_path'], config
    )
    xgb_pred = xgb_model.predict(xgb.DMatrix(xgb_X_val))
    xgb_rmse = mean_squared_error(xgb_y_val, xgb_pred, squared=False)
    logging.info(f'XGBoost RMSE: {xgb_rmse}')
    
    # Train LightGBM
    lgbm_model, lgbm_X_val, lgbm_y_val = train_lightgbm(
        config['data']['train_lgbm_path'], config
    )
    lgbm_pred = lgbm_model.predict(lgbm_X_val)
    lgbm_rmse = mean_squared_error(lgbm_y_val, lgbm_pred, squared=False)
    logging.info(f'LightGBM RMSE: {lgbm_rmse}')
    
    # Train CatBoost
    cat_model, cat_X_val, cat_y_val = train_catboost(
        config['data']['train_processed_path'], config
    )
    cat_pred = cat_model.predict(cat_X_val)
    cat_rmse = mean_squared_error(cat_y_val, cat_pred, squared=False)
    logging.info(f'CatBoost RMSE: {cat_rmse}')
    
    # Save models
    xgb_model.save_model(config['models']['xgboost']['model_path'])
    lgbm_model.save_model(config['models']['lightgbm']['model_path'])
    cat_model.save_model(config['models']['catboost']['model_path'])
    
    print("Model training complete.")

if __name__ == "__main__":
    main()
