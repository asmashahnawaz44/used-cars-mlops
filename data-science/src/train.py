# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path to save trained model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()
    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read data
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")

    # Split features and target
    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    # Hyperparameter tuning - try multiple combinations and log each to MLflow
    param_grid = [
        {"n_estimators": 10, "max_depth": 3},
        {"n_estimators": 20, "max_depth": 5},
        {"n_estimators": 50, "max_depth": 7},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 7},
    ]

    best_mse = float("inf")
    best_model = None
    best_params = None

    for params in param_grid:
        with mlflow.start_run(nested=True):
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Log params and metrics
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_metric("MSE", mse)

            print(f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, MSE={mse}")

            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = params

    print(f"\nBest params: {best_params}, Best MSE: {best_mse}")

    # Log best params and metrics
    mlflow.log_param("best_n_estimators", best_params["n_estimators"])
    mlflow.log_param("best_max_depth", best_params["max_depth"])
    mlflow.log_metric("best_MSE", best_mse)

    # Save best model
    mlflow.sklearn.save_model(best_model, args.model_output)
    print(f"Best model saved to: {args.model_output}")


if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]
    for line in lines:
        print(line)

    main(args)
    mlflow.end_run()
