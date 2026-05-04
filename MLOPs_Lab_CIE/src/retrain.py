import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

df1 = pd.read_csv("data/training_data.csv")
df2 = pd.read_csv("data/new_data.csv")

combined = pd.concat([df1, df2])

X = combined.drop("lap_time_seconds", axis=1)
y = combined["lap_time_seconds"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

retrained_rmse = np.sqrt(mean_squared_error(y_test, preds))

champion_rmse = 4.5  # from step1

improvement = champion_rmse - retrained_rmse

action = "promoted" if improvement >= 1.0 else "kept_champion"

output = {
    "original_data_rows": len(df1),
    "new_data_rows": len(df2),
    "combined_data_rows": len(combined),
    "champion_rmse": champion_rmse,
    "retrained_rmse": retrained_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 1.0,
    "action": action,
    "comparison_metric": "rmse"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)