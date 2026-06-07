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

def find_model_path(base_path):
    '''Find the actual model path'''
    if os.path.exists(os.path.join(base_path, "MLmodel")):
        return base_path
    for root, dirs, files in os.walk(base_path):
        if "MLmodel" in files:
            return root
    return base_path

def main(args):
    '''Loads the best-trained model and registers it'''

    print("Registering ", args.model_name)
    print("Model path: ", args.model_path)

    if os.path.exists(args.model_path):
        print("Contents of model path:")
        for item in os.listdir(args.model_path):
            print(f"  {item}")

    actual_model_path = find_model_path(args.model_path)
    print(f"Actual model path: {actual_model_path}")

    # Step 1: Load the model
    model = mlflow.sklearn.load_model(actual_model_path)

    # Step 2: End any active run first
    mlflow.end_run()

    # Step 3: Log and register the model in a new run
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model,
            artifact_path=args.model_name
        )
        run_id = mlflow.active_run().info.run_id

    # Step 4: Register the model
    model_uri = f"runs:/{run_id}/{args.model_name}"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name
    )

    # Step 5: Write model info to JSON
    os.makedirs(args.model_info_output_path, exist_ok=True)
    model_info = {
        "model_name": registered_model.name,
        "model_version": registered_model.version
    }
    with open(Path(args.model_info_output_path) / "model_info.json", "w") as f:
        json.dump(model_info, f)

    print(f"Model registered: {registered_model.name}, version: {registered_model.version}")


if __name__ == "__main__":

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
