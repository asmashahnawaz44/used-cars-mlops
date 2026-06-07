# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

    # Step 1: Load the model from the specified path
    model = mlflow.sklearn.load_model(args.model_path)

    # Step 2: Log the loaded model in MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model,
            artifact_path=args.model_name
        )
        run_id = mlflow.active_run().info.run_id

    # Step 3: Register the logged model
    model_uri = f"runs:/{run_id}/{args.model_name}"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name
    )

    # Step 4: Write model registration details to JSON file
    os.makedirs(args.model_info_output_path, exist_ok=True)
    model_info = {
        "model_name": registered_model.name,
        "model_version": registered_model.version
    }
    with open(Path(args.model_info_output_path) / "model_info.json", "w") as f:
        json.dump(model_info, f)

    print(f"Model registered: {registered_model.name}, version: {registered_model.version}")


if __name__ == "__main__":

    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
