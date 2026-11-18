"""Preparación de datos a partir de IEEE-CIS Fraud Detection (Kaggle).

Transforma el dataset transaccional en los insumos del caso real:

1. **Base Histórica** (días < history_cutoff_day): perfiles por cliente y listas de bloqueo.
2. **Reportes de corte** (días posteriores, lunes a viernes, cortes 09:00 / 12:00 / 16:00):
   CSV que simulan los archivos que envía la plataforma de monitoreo. Un cliente entra al
   reporte cuando su volumen dentro de la ventana del corte supera los umbrales.
3. **Ground truth**: etiquetas isFraud por transacción, guardadas aparte. Los agentes nunca
   las ven; sólo se usan para la evaluación (matriz de confusión / KPIs).

Anonimización: los identificadores (cliente, tarjeta, dispositivo) se reemplazan por SHA-256.

Uso:
    python -m fraudguard.data_prep
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from fraudguard import config, risk_model

TX_COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "dist1",
    "P_emaildomain", "D1",
]
ID_COLS = ["TransactionID", "DeviceType", "DeviceInfo"]

# Ventanas de cada corte: (hora_inicio, hora_fin, día_inicio_relativo)
VENTANAS = {
    "09:00": (16, 9, -1),   # 16:00 del día anterior -> 09:00
    "12:00": (9, 12, 0),
    "16:00": (12, 16, 0),
}

REPORT_COLUMNS = [
    "id_alerta", "corte", "transaction_id", "fecha_hora", "cliente_id", "tarjeta_id",
    "monto", "producto", "tipo_tarjeta", "red_tarjeta", "region", "distancia",
    "dominio_email", "tipo_dispositivo", "dispositivo_id",
]


def sha(value: str) -> str:
    return hashlib.sha256(f"{config.HASH_SALT}|{value}".encode()).hexdigest()[:16]


def _hash_series(s: pd.Series) -> pd.Series:
    uniq = s.dropna().unique()
    mapping = {v: sha(str(v)) for v in uniq}
    return s.map(mapping)


def load_raw() -> pd.DataFrame:
    model_cols = [c for c in risk_model.FEATURES if c not in ("DeviceType",)]
    cols = list(dict.fromkeys(TX_COLS + model_cols))
    f32 = {c: "float32" for c in risk_model.NUM_COLS if c not in ("TransactionAmt", "card1", "addr1", "D1")}
    tx = pd.read_csv(config.RAW_DIR / "train_transaction.csv", usecols=cols, dtype=f32)
    idf = pd.read_csv(config.RAW_DIR / "train_identity.csv", usecols=ID_COLS)
    df = tx.merge(idf, on="TransactionID", how="left").copy()
    base = pd.Timestamp(config.rules()["dataset"]["base_date"])
    df["fecha_hora"] = base + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["day"] = (df["TransactionDT"] // 86400).astype(int)
    # Identificador de cliente (técnica conocida en la competencia): tarjeta + región + fecha
    # de apertura implícita (día - D1). Aproxima mejor a "un cliente" que card1 solo.
    d1n = (df["day"] - df["D1"].fillna(-1)).astype(int)
    df["uid_raw"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + d1n.astype(str)
    df["card_raw"] = (
        df["card1"].astype(str) + "_" + df["card2"].astype(str) + "_"
        + df["card4"].astype(str) + "_" + df["card6"].astype(str)
    )
    df["cliente_id"] = _hash_series(df["uid_raw"])
    df["tarjeta_id"] = _hash_series(df["card_raw"])
    df["dispositivo_id"] = _hash_series(df["DeviceInfo"])
    return df


def build_history(hist: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Perfiles por cliente + listas de bloqueo, calculados sólo con la Base Histórica."""
    cfg = config.rules()["listas_bloqueo"]
    daily = hist.groupby(["cliente_id", "day"]).agg(n=("TransactionID", "size"))
    prof_daily = daily.groupby("cliente_id")["n"].agg(
        dias_activos="size", tx_diarias_media="mean", tx_diarias_std="std"
    )
    prof_amt = hist.groupby("cliente_id").agg(
        tx_total=("TransactionID", "size"),
        monto_medio=("TransactionAmt", "mean"),
        monto_std=("TransactionAmt", "std"),
        monto_max=("TransactionAmt", "max"),
        ultima_fecha=("fecha_hora", "max"),
    )
    profiles = prof_daily.join(prof_amt).reset_index()
    profiles[["tx_diarias_std", "monto_std"]] = profiles[["tx_diarias_std", "monto_std"]].fillna(0.0)

    fraud = hist[hist["isFraud"] == 1]
    dev = hist.dropna(subset=["dispositivo_id"]).groupby("dispositivo_id")["isFraud"].agg(["size", "mean"])
    dev_block = dev[(dev["size"] >= cfg["dispositivo_min_tx"]) & (dev["mean"] >= cfg["dispositivo_min_tasa_fraude"])]
    tasa_global = hist["isFraud"].mean()
    em = hist.dropna(subset=["P_emaildomain"]).groupby("P_emaildomain")["isFraud"].agg(["size", "mean"])
    em_risk = em[(em["size"] >= cfg["email_min_tx"]) & (em["mean"] >= cfg["email_factor_tasa_global"] * tasa_global)]

    # card1..card6 identifica un grupo de tarjetas (no una tarjeta única): sólo se bloquean
    # grupos con tasa de fraude alta, igual que los dispositivos.
    card = hist.dropna(subset=["tarjeta_id"]).groupby("tarjeta_id")["isFraud"].agg(["size", "mean"])
    card_block = card[(card["size"] >= cfg["tarjeta_min_tx"]) & (card["mean"] >= cfg["tarjeta_min_tasa_fraude"])]

    blocklists = {
        "clientes": sorted(fraud["cliente_id"].dropna().unique().tolist()),
        "tarjetas": sorted(card_block.index.tolist()),
        "dispositivos": sorted(dev_block.index.tolist()),
        "dominios_email_riesgo": sorted(em_risk.index.tolist()),
        "tasa_fraude_global_historica": float(tasa_global),
    }
    return profiles, blocklists


def _window_bounds(day_ts: pd.Timestamp, corte: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    h_ini, h_fin, d_rel = VENTANAS[corte]
    ini = day_ts + pd.Timedelta(days=d_rel, hours=h_ini)
    fin = day_ts + pd.Timedelta(hours=h_fin)
    return ini, fin


def build_reports(sim: pd.DataFrame) -> pd.DataFrame:
    """Genera un CSV por corte hábil. Devuelve el índice de reportes generados."""
    cfg = config.rules()["alert_generation"]
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(sim["fecha_hora"].min().normalize() + pd.Timedelta(days=1),
                         sim["fecha_hora"].max().normalize(), freq="D")
    index_rows = []
    for day_ts in days:
        if day_ts.weekday() >= 5:  # sólo lunes a viernes
            continue
        for corte in config.CORTES:
            ini, fin = _window_bounds(day_ts, corte)
            w = sim[(sim["fecha_hora"] >= ini) & (sim["fecha_hora"] < fin)]
            if w.empty:
                continue
            g = w.groupby("cliente_id").agg(n=("TransactionID", "size"), s=("TransactionAmt", "sum"))
            flagged = g[(g["n"] >= cfg["min_tx_count"]) | (g["s"] >= cfg["min_amount_sum"])].index
            rep = w[w["cliente_id"].isin(flagged)].copy()
            if rep.empty:
                continue
            report_id = f"ALR-{day_ts:%Y%m%d}-{corte.replace(':', '')}"
            out = pd.DataFrame({
                "id_alerta": report_id,
                "corte": corte,
                "transaction_id": rep["TransactionID"].astype(int),
                "fecha_hora": rep["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                "cliente_id": rep["cliente_id"],
                "tarjeta_id": rep["tarjeta_id"],
                "monto": rep["TransactionAmt"].round(2),
                "producto": rep["ProductCD"],
                "tipo_tarjeta": rep["card6"],
                "red_tarjeta": rep["card4"],
                "region": rep["addr1"],
                "distancia": rep["dist1"],
                "dominio_email": rep["P_emaildomain"],
                "tipo_dispositivo": rep["DeviceType"],
                "dispositivo_id": rep["dispositivo_id"],
            }).sort_values("fecha_hora")
            fname = f"alerta_{day_ts:%Y%m%d}_{corte.replace(':', '')}.csv"
            out.to_csv(config.REPORTS_DIR / fname, index=False)
            index_rows.append({
                "archivo": fname, "id_alerta": report_id, "fecha": f"{day_ts:%Y-%m-%d}",
                "semana_iso": f"{day_ts.isocalendar().year}-W{day_ts.isocalendar().week:02d}",
                "corte": corte, "n_transacciones": len(out), "n_clientes": len(flagged),
            })
    return pd.DataFrame(index_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cutoff = config.rules()["dataset"]["history_cutoff_day"]

    print("Cargando datos crudos...")
    df = load_raw()
    hist, sim = df[df["day"] < cutoff], df[df["day"] >= cutoff]
    print(f"  Base Histórica: {len(hist):,} tx | Periodo simulado: {len(sim):,} tx")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    print("Construyendo perfiles y listas de bloqueo...")
    profiles, blocklists = build_history(hist)
    profiles.to_parquet(config.PROCESSED_DIR / "perfiles_clientes.parquet", index=False)
    with open(config.PROCESSED_DIR / "listas_bloqueo.json", "w", encoding="utf-8") as f:
        json.dump(blocklists, f, indent=1)
    print(f"  {len(profiles):,} perfiles | bloqueo: {len(blocklists['clientes'])} clientes, "
          f"{len(blocklists['tarjetas'])} tarjetas, {len(blocklists['dispositivos'])} dispositivos, "
          f"{len(blocklists['dominios_email_riesgo'])} dominios de riesgo")

    print("Entrenando modelo de riesgo con la Base Histórica (validación temporal)...")
    print(f"  {risk_model.train(hist)}")
    risk_model.save_detail(sim)

    print("Generando reportes de corte (09:00 / 12:00 / 16:00)...")
    idx = build_reports(sim)
    idx.to_csv(config.REPORTS_DIR / "_indice_reportes.csv", index=False)
    sim[["TransactionID", "isFraud", "cliente_id"]].rename(columns={"TransactionID": "transaction_id"}).to_parquet(
        config.GROUND_TRUTH_DIR / "etiquetas_transacciones.parquet", index=False
    )
    print(f"  {len(idx)} reportes | semanas: {idx['semana_iso'].nunique()} | "
          f"clientes por reporte (mediana): {int(np.median(idx['n_clientes']))}")


if __name__ == "__main__":
    main()
