# src/models/hyperparameter_tuning.py

import pandas as pd
import numpy as np
import optuna
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
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

def tune_xgboost(train_path, config):
    """Hyperparameter tuning for XGBoost using Optuna."""
    train = pd.read_csv(train_path)
    X = train.drop(['id', 'p_num', 'bg+1:00'], axis=1)
    y = train['bg+1:00']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.2),
            'max_depth': trial.suggest_int('max_depth', 10, 31),
            'gamma': trial.suggest_float('gamma', 0.1, 0.5),
            'subsample': trial.suggest_float('subsample', 0.7, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.8),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.932),
            'device': 'cuda',
            'verbosity': 0
        }
        num_rounds = config['xgboost']['num_rounds']
        model = xgb.train(params, dtrain, num_rounds, evals=[(dval, 'eval')], verbose_eval=False)
        preds = model.predict(dval)
        mse = mean_squared_error(y_val, preds)
        return mse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=config['xgboost']['optuna_trials'])
    logging.info(f'XGBoost Best Params: {study.best_params}')
    logging.info(f'XGBoost Best MSE: {study.best_value}')
    return study.best_params, study.best_value

def tune_lightgbm(train_lgbm_path, config):
    """Hyperparameter tuning for LightGBM using Optuna."""
    train = pd.read_csv(train_lgbm_path)
    X = train.drop(['id', 'p_num', 'bg_1hour'], axis=1)
    y = train['bg_1hour']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'num_leaves': trial.suggest_int('num_leaves', 58, 70),
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.3, 0.4),
            'subsample': trial.suggest_float('subsample', 0.38, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.66),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 0.17),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.19)
        }
        num_rounds = config['lightgbm']['num_rounds']
        model = lgb.train(params, lgb_train, num_rounds, valid_sets=[lgb_eval], verbose_eval=False)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=config['lightgbm']['optuna_trials'])
    logging.info(f'LightGBM Best Params: {study.best_params}')
    logging.info(f'LightGBM Best RMSE: {study.best_value}')
    return study.best_params, study.best_value

def tune_catboost(train_path, config):
    """Hyperparameter tuning for CatBoost using Hyperopt."""
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    
    train = pd.read_csv(train_path)
    X = train.drop(['id', 'p_num', 'bg+1:00'], axis=1)
    y = train['bg+1:00']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    def objective(params):
        model = CatBoostRegressor(
            task_type="GPU",
            learning_rate=params['learning_rate'],
            depth=params['depth'],
            l2_leaf_reg=params['l2_leaf_reg'],
            iterations=params['iterations'],
            verbose=False
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        return {'loss': rmse, 'status': STATUS_OK}
    
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'depth': hp.randint('depth', 3, 10),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 20),
        'iterations': hp.randint('iterations', 100, 1000)
    }
    
    results_cat = []
    for _ in range(5):
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=config['catboost']['hyperopt_evals'],
            trials=trials
        )
        results_cat.append(best)
        logging.info(f'CatBoost Best Params: {best}')
    
    return results_cat

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = load_config()
    
    # Tune XGBoost
    xgb_best_params, xgb_best_mse = tune_xgboost(
        config['data']['train_processed_path'], config
    )
    
    # Tune LightGBM
    lgbm_best_params, lgbm_best_rmse = tune_lightgbm(
        config['data']['train_lgbm_path'], config
    )
    
    # Tune CatBoost
    catboost_best = tune_catboost(
        config['data']['train_processed_path'], config
    )
    
    # Save tuned parameters
    with open(config['models']['xgboost']['tuned_params_path'], 'w') as f:
        yaml.dump(xgb_best_params, f)
    with open(config['models']['lightgbm']['tuned_params_path'], 'w') as f:
        yaml.dump(lgbm_best_params, f)
    with open(config['models']['catboost']['tuned_params_path'], 'w') as f:
        yaml.dump(catboost_best, f)
    
    print("Hyperparameter tuning complete.")

if __name__ == "__main__":
    main()
