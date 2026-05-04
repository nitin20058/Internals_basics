import mlflow
import json

model_uri = "runs:/83366612e3564c909657d4a22e2ad193/GradientBoosting"

result = mlflow.register_model(
    model_uri,
    "swimsync-lap-time-seconds-predictor"
)

output = {
    "registered_model_name": "swimsync-lap-time-seconds-predictor",
    "version": result.version,
    "run_id": result.run_id,
    "source_metric": "rmse",
    "source_metric_value":  5.954076629512856
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)