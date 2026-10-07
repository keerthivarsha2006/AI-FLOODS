from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import MODEL_FEATURES


MODEL_TARGET = "flood_risk"


def build_model() -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
            )),
        ]
    )
    return model


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = df[MODEL_FEATURES].copy()
    target = df[MODEL_TARGET].astype(int)
    return feature_frame, target


def train_and_evaluate(df: pd.DataFrame) -> dict[str, Any]:
    X, y = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)) if len(np.unique(y_test)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    return {"model": model, "metrics": metrics}


def score_observation(observation: dict[str, Any], model: Any) -> tuple[float, str, dict[str, Any]]:
    row = pd.DataFrame([observation])
    feature_row = row[MODEL_FEATURES].copy()
    risk_score = float(model.predict_proba(feature_row)[0, 1] * 100)
    if risk_score >= 75:
        level = "CRITICAL"
    elif risk_score >= 50:
        level = "HIGH"
    elif risk_score >= 25:
        level = "MEDIUM"
    else:
        level = "LOW"

    factors = {
        "Heavy rainfall": float(observation.get("rain_1h", 0.0)) > 10,
        "Rapid water-level rise": abs(float(observation.get("rate_of_rise", 0.0))) > 0.3,
        "High water level": float(observation.get("water_level", 0.0)) >= float(observation.get("warning_level", 0.0)),
        "Forecast rainfall increase": float(observation.get("forecast_3h", 0.0)) > float(observation.get("forecast_1h", 0.0)),
        "Low elevation": float(observation.get("elevation_m", 0.0)) < 100,
    }
    primary = [name for name, flag in factors.items() if flag]
    if not primary:
        primary = ["Persistent wet conditions"]
    top_factors = primary[:3]
    explanation = {
        "risk_score": risk_score,
        "level": level,
        "factors": top_factors,
    }
    return risk_score, level, explanation


def generate_future_risk(current_score: float) -> dict[str, float]:
    return {
        "now": float(current_score),
        "+1h": float(min(100.0, current_score + 8)),
        "+3h": float(min(100.0, current_score + 12)),
        "+6h": float(min(100.0, current_score + 18)),
        "+12h": float(min(100.0, current_score + 24)),
        "+24h": float(min(100.0, current_score + 30)),
    }
