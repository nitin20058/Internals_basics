import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv("data/training_data.csv")

X = df.drop("lap_time_seconds", axis=1)
y = df["lap_time_seconds"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("swimsync-lap-time-seconds")

models = {
    "Lasso": Lasso(alpha=0.1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = []

best_rmse = float("inf")
best_model_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape_val = mape(y_test, preds)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape_val
        })
        mlflow.set_tag("experiment_type", "baseline_comparison")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape_val
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

        # save best model
        import joblib
        joblib.dump(model, f"models/{name}.pkl")

output = {
    "experiment_name": "swimsync-lap-time-seconds",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)