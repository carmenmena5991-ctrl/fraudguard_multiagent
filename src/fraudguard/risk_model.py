"""Modelo de riesgo transaccional (herramienta determinista del Agente 2).

HistGradientBoosting entrenado SÓLO con la Base Histórica (días < history_cutoff_day) sobre las
variables anonimizadas de IEEE-CIS (C, D, M, V, monto, producto, tarjeta, email, dispositivo).
Se valida con split temporal (entrena días < cutoff-20, valida en los últimos 20 días del histórico).

El Agente 2 lo aplica sobre el detalle transaccional (`detalle_transaccional.parquet`) consultado
por transaction_id, igual que el analista consulta el dashboard de micro-transacciones.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudguard import config

CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType",
            *[f"M{i}" for i in range(1, 10)]]
NUM_COLS = (["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1"]
            + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
            + [f"V{i}" for i in [*range(12, 138), *range(279, 322)]])
FEATURES = NUM_COLS + CAT_COLS
MODEL_PATH = config.PROCESSED_DIR / "modelo_riesgo.joblib"
DETAIL_PATH = config.PROCESSED_DIR / "detalle_transaccional.parquet"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X[NUM_COLS] = X[NUM_COLS].astype("float32")
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          categorical_features="from_dtype", random_state=42)


def train(hist: pd.DataFrame) -> dict:
    """Entrena con validación temporal y luego reentrena con todo el histórico."""
    cutoff = config.rules()["dataset"]["history_cutoff_day"]
    X, y = prepare(hist), hist["isFraud"].to_numpy()
    tr = (hist["day"] < cutoff - 20).to_numpy()
    m = _model().fit(X[tr], y[tr])
    p = m.predict_proba(X[~tr])[:, 1]
    metrics = {"val_auc": round(float(roc_auc_score(y[~tr], p)), 4),
               "val_avg_precision": round(float(average_precision_score(y[~tr], p)), 4),
               "n_train": int(tr.sum()), "n_val": int((~tr).sum())}
    final = _model().fit(X, y)
    # Las categorías vistas en entrenamiento se guardan para alinear datos nuevos.
    cats = {c: X[c].cat.categories.tolist() for c in CAT_COLS}
    joblib.dump({"model": final, "categories": cats, "metrics": metrics}, MODEL_PATH)
    return metrics


def save_detail(sim: pd.DataFrame) -> None:
    """Detalle transaccional del periodo simulado (sin la etiqueta isFraud)."""
    cols = ["TransactionID", *FEATURES]
    sim[cols].rename(columns={"TransactionID": "transaction_id"}).to_parquet(DETAIL_PATH, index=False)


@lru_cache(maxsize=1)
def _load():
    bundle = joblib.load(MODEL_PATH)
    detail = pd.read_parquet(DETAIL_PATH).set_index("transaction_id")
    return bundle, detail


def disponible() -> bool:
    return MODEL_PATH.exists() and DETAIL_PATH.exists()


def score_transacciones(transaction_ids: list[int]) -> pd.Series:
    """Probabilidad de fraude por transacción. NaN si la transacción no está en el detalle."""
    bundle, detail = _load()
    ids = pd.Index(transaction_ids)
    found = ids[ids.isin(detail.index)]
    out = pd.Series(np.nan, index=ids, dtype=float)
    if len(found):
        X = detail.loc[found, FEATURES].copy()
        X[NUM_COLS] = X[NUM_COLS].astype("float32")
        for c in CAT_COLS:
            X[c] = pd.Categorical(X[c], categories=bundle["categories"][c])
        out.loc[found] = bundle["model"].predict_proba(X)[:, 1]
    return out


def metricas_modelo() -> dict:
    return joblib.load(MODEL_PATH)["metrics"] if MODEL_PATH.exists() else {}


if __name__ == "__main__":
    print(json.dumps(metricas_modelo(), indent=2))
