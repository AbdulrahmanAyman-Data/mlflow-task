import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "california-housing-price-prediction"

RANDOM_SEED = 42       
VALIDATION_SIZE = 0.2  
N_ESTIMATORS = 100     

RUNS = [
    {"name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]


def load_and_split():
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target
    return train_test_split(X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_one_run(config, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=config["name"]) as run:
        # Parameters
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.log_param("learning_rate", config["learning_rate"])
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("random_seed", RANDOM_SEED)

        # Train
        model = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        # Validation metrics
        predictions = model.predict(X_val)
        metrics = evaluate(y_val, predictions)
        mlflow.log_metrics(metrics)

        # Model artifact
        signature = infer_signature(X_val, predictions)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_val.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        print(
            f"{config['name']}: max_depth={config['max_depth']}, "
            f"learning_rate={config['learning_rate']} -> "
            f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}"
        )
        return run.info.run_id


def compare_runs(client, run_ids):
    rows = []
    for run_id in run_ids:
        run = client.get_run(run_id)
        rows.append({
            "run_id": run_id,
            "name": run.info.run_name,
            "max_depth": run.data.params["max_depth"],
            "learning_rate": run.data.params["learning_rate"],
            "rmse": run.data.metrics["rmse"],
            "mae": run.data.metrics["mae"],
            "r2": run.data.metrics["r2"],
        })

    best = min(rows, key=lambda row: row["rmse"])  # lower RMSE is better
    client.set_tag(best["run_id"], "best_model", "true")

    print("\n| Run | Max Depth | Learning Rate | RMSE | MAE | R2 |")
    print("|-----|-----------|---------------|------|-----|----|")
    for row in rows:
        print(
            f"| {row['name']} | {row['max_depth']} | {row['learning_rate']} | "
            f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |"
        )
    print(
        f"\nBest model (lowest RMSE): {best['name']} "
        f"(max_depth={best['max_depth']}, learning_rate={best['learning_rate']}, "
        f"RMSE={best['rmse']:.4f}, run_id={best['run_id']})"
    )
    return best


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, y_train, y_val = load_and_split()
    print(f"Training rows: {len(X_train)}  Validation rows: {len(X_val)}\n")

    run_ids = [train_one_run(config, X_train, X_val, y_train, y_val) for config in RUNS]
    compare_runs(MlflowClient(), run_ids)


if __name__ == "__main__":
    main()
