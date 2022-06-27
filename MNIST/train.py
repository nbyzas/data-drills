from pathlib import Path

import numpy as np
import mlflow
mlflow.autolog()

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_mastor import mastor as ms

mnist = fetch_openml(
    "mnist_784", version=1, cache=True, as_frame=False, data_home="./skdata"
)


X = mnist["data"]
y = mnist["target"].astype(np.uint8)


X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]


mlflow.set_tracking_uri("sqlite:////home/nikley/Projects/dataset-drilling/MNIST/mlflow_backend.db")

run_name = "berb"
with mlflow.start_run(run_name=run_name) as run:
    artifact_uri = run.info.artifact_uri
    model_path = str(Path(artifact_uri) / Path("model"))
    print(f"Run id: {run.info.run_id}")

    # Define model
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", SVC())]
    )
    parameters = [
        {
            "classifier__kernel": ["linear", "poly", "rbf"],
            "classifier__C": [0.5, 1, 2],
        },
    ]
    model = GridSearchCV(
        pipeline, parameters, cv=2, scoring="f1_macro", n_jobs=8
    )


    print("TRAINING")
    ind = ms.subset_indices(y_train, ratio=0.025)
    X_train_subset = X_train[ind]
    y_train_subset = y_train[ind]
    print(f"Num training samples: {len(y_train_subset)}")
    model.fit(X_train_subset, y_train_subset)