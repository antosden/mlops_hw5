import pickle
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yaml


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("HW5 MLOps")


with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

model_params = params["model"]

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

target = "variety"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

with mlflow.start_run(run_name="iris_random_forest"):
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    signature = infer_signature(X_test, y_pred)

    mlflow.log_params(model_params)
    mlflow.log_metric("accuracy", accuracy)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.log_artifact("model.pkl")
    mlflow.sklearn.log_model(
        sk_model=model,
        name="sklearn-model",
        signature=signature,
        registered_model_name="iris-random-forest-model",
    )

    print(f"Accuracy: {accuracy}")