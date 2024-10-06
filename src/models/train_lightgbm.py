import mlflow
import mlflow.xgboost
import xgboost as xgb

def train_lightgbm(X_train, y_train):
    # Define your model
    model = xgb.XGBClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Log the model
    mlflow.xgboost.log_model(model, "model")

    return model
