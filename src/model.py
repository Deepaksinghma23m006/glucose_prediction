# src/model.py
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import mlflow

def train_model(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'learning_rate': 0.1,
        'max_depth': 6,
    }

    with mlflow.start_run():
        model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')], verbose_eval=False)
        y_pred = model.predict(dval)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_params(params)
        mlflow.log_metric('rmse', rmse)
        mlflow.xgboost.log_model(model, "model")
        
        return model, rmse