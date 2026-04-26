import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# подключаем MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("HW5 MLOps")

# загружаем данные
df = pd.read_csv("energydata_complete_v1.csv")

target = "variety"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
}

with mlflow.start_run():
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # логируем
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)