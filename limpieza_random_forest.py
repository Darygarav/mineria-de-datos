from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DATA_URL = "https://raw.githubusercontent.com/Darygarav/base-de-datos-australia/main/weatherAUS.csv"
LOCAL_DATASET = Path("weatherAUS.csv")
TARGET = "RainTomorrow"

# Variables con demasiados nulos para esta primera version del modelo.
HIGH_MISSING_COLUMNS = ["Sunshine", "Evaporation", "Cloud3pm", "Cloud9am"]

# Variable con fuga de informacion: mide lluvia/riesgo del dia siguiente.
LEAKAGE_COLUMNS = ["RISK_MM"]


@dataclass
class RainTomorrowArtifacts:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    model: Pipeline


def load_weather_data(url: str = DATA_URL) -> pd.DataFrame:
    """
    Carga el dataset meteorologico.

    Prioriza un CSV local para facilitar pruebas sin conexion.
    """
    if LOCAL_DATASET.exists():
        return pd.read_csv(LOCAL_DATASET)
    return pd.read_csv(url)


def clean_weather_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Limpia el dataset para predecir RainTomorrow.

    Decisiones principales:
    - elimina columnas con muchos nulos;
    - elimina la fuga de informacion RISK_MM;
    - transforma Date en variables utiles;
    - convierte la variable objetivo a binaria.
    """
    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data = data.drop(columns=["Date"])

    drop_columns = [col for col in HIGH_MISSING_COLUMNS + LEAKAGE_COLUMNS if col in data.columns]
    data = data.drop(columns=drop_columns)

    data = data[data[TARGET].notna()].copy()
    data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1})
    data = data[data[TARGET].notna()].copy()
    data[TARGET] = data[TARGET].astype(int)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Crea el bloque de imputacion y codificacion."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_decision_tree_pipeline(X: pd.DataFrame) -> Pipeline:
    """Crea el pipeline completo para entrenamiento y datos futuros."""
    preprocessor = build_preprocessor(X)

    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_training_artifacts(df: pd.DataFrame) -> RainTomorrowArtifacts:
    """Genera train/test listos para entrenar y evaluar."""
    X, y = clean_weather_data(df)

    class_counts = y.value_counts()
    test_size = 0.2
    estimated_test_rows = max(1, int(round(len(y) * test_size)))
    use_stratify = (
        len(class_counts) > 1
        and class_counts.min() >= 2
        and estimated_test_rows >= len(class_counts)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y if use_stratify else None,
        random_state=42,
    )

    pipeline = build_decision_tree_pipeline(X_train)
    return RainTomorrowArtifacts(X_train, X_test, y_train, y_test, pipeline)


def main() -> None:
    df = load_weather_data()
    source_name = str(LOCAL_DATASET) if LOCAL_DATASET.exists() else DATA_URL
    artifacts = prepare_training_artifacts(df)
    artifacts.model.fit(artifacts.X_train, artifacts.y_train)

    print("Limpieza completada para RainTomorrow")
    print(f"Fuente de datos: {source_name}")
    print(f"Filas entrenamiento: {len(artifacts.X_train):,}")
    print(f"Filas prueba: {len(artifacts.X_test):,}")
    print(f"Variables de entrada: {artifacts.X_train.shape[1]}")
    print("\nColumnas usadas:")
    for column in artifacts.X_train.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()
