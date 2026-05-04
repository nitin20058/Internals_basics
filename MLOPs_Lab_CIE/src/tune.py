import pandas as pd
import json
import mlflow
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/training_data.csv")

X = df.drop("lap_time_seconds", axis=1)
y = df["lap_time_seconds"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5]
}

mlflow.set_experiment("swimsync-lap-time-seconds")

with mlflow.start_run(run_name="tuning-swimsync") as parent_run:

    model = GradientBoostingRegressor(random_state=42)

    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=5,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    output = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": 5,
        "best_params": search.best_params_,
        "best_mae": mae,
        "best_cv_mae": -search.best_score_,
        "parent_run_name": "tuning-swimsync"
    }

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)