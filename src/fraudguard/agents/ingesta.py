"""Agente 1 - Ingesta y Segmentación.

Valida integridad del archivo de corte (esquema determinista, sin LLM), remueve PII,
y segmenta cada caso (cliente) en Alto / Medio / Bajo. Si el archivo no es un reporte
transaccional válido, detiene el flujo y solicita recarga al analista.
"""
from __future__ import annotations

import hashlib
import io
import re

import pandas as pd

from fraudguard import config
from fraudguard.agents.base import AgentError, BaseAgent
from fraudguard.schemas import CasoSegmentado, IngestaInput, IngestaOutput, Transaccion

REQUIRED = ["id_alerta", "corte", "transaction_id", "fecha_hora", "cliente_id", "monto"]
OPTIONAL = ["tarjeta_id", "producto", "tipo_tarjeta", "red_tarjeta", "region", "distancia",
            "dominio_email", "tipo_dispositivo", "dispositivo_id"]
# Columnas que podrían traer PII en un export real: se eliminan o se hashean.
PII_DROP = ["nombre", "nombre_cliente", "email", "telefono", "direccion", "numero_tarjeta", "pan", "cuenta"]
HEX16 = re.compile(r"^[0-9a-f]{16,64}$")
SEG_ORDER = {"Bajo": 0, "Medio": 1, "Alto": 2}


class IngestaAgent(BaseAgent[IngestaInput, IngestaOutput]):
    nombre = "Agente 1 - Ingesta y Segmentación"
    input_model = IngestaInput
    output_model = IngestaOutput
    herramientas_permitidas = ("pandas.read_csv", "validador_esquema", "anonimizador_sha256", "segmentador_riesgo")

    def _loggable_input(self, data: IngestaInput):
        return {"nombre_archivo": data.nombre_archivo, "corte_declarado": data.corte_declarado,
                "bytes": len(data.contenido_csv)}

    def _run(self, data: IngestaInput) -> IngestaOutput:
        if not data.nombre_archivo.lower().endswith(".csv"):
            raise AgentError("El archivo no es un CSV de reporte transaccional. Cargue el reporte del corte.")
        self.use_tool("pandas.read_csv")
        try:
            df = pd.read_csv(io.BytesIO(data.contenido_csv))
        except Exception as e:
            raise AgentError(f"Archivo corrupto o ilegible: {e}. Solicite recarga del reporte.") from e

        self.use_tool("validador_esquema")
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise AgentError(f"Reporte incompleto: faltan columnas obligatorias {missing}. "
                             "Verifique si la plataforma de monitoreo cambió el formato.")
        if df.empty:
            raise AgentError("El reporte no contiene transacciones.")
        if df["id_alerta"].nunique() != 1 or df["corte"].nunique() != 1:
            raise AgentError("El archivo mezcla varias alertas o cortes; debe contener un único corte.")
        corte = str(df["corte"].iloc[0])
        if corte not in config.CORTES:
            raise AgentError(f"Corte '{corte}' inválido; se esperan {config.CORTES}.")
        if data.corte_declarado and data.corte_declarado != corte:
            raise AgentError(f"El analista declaró el corte {data.corte_declarado} pero el archivo es del {corte}.")
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
        df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
        bad = df["fecha_hora"].isna() | df["monto"].isna() | df["cliente_id"].isna() | (df["monto"] < 0)
        if bad.mean() > 0.05:
            raise AgentError(f"{int(bad.sum())} filas con fecha, monto o cliente inválidos (>5%). Solicite recarga.")
        df = df[~bad]
        if df["transaction_id"].duplicated().any():
            df = df.drop_duplicates("transaction_id")

        self.use_tool("anonimizador_sha256")
        pii = [c for c in df.columns if c in PII_DROP]
        df = df.drop(columns=pii)
        for col in ("cliente_id", "tarjeta_id", "dispositivo_id"):
            if col in df.columns:
                crudo = df[col].map(lambda v: not pd.isna(v) and not HEX16.match(str(v)))
                if crudo.any():  # identificadores sin anonimizar -> SHA-256
                    df.loc[crudo, col] = df.loc[crudo, col].map(_sha)
                    pii.append(f"{col} (hasheado)")
        for c in OPTIONAL:
            if c not in df.columns:
                df[c] = None

        self.use_tool("segmentador_riesgo")
        df["segmento_tx"] = segmentar_tx(df)
        casos = []
        for cid, g in df.groupby("cliente_id", sort=False):
            seg = max(g["segmento_tx"], key=SEG_ORDER.get)
            seg = _escalar_por_volumen(seg, len(g))
            txs = [Transaccion(**_clean(r)) for r in g[["transaction_id", "fecha_hora", "cliente_id", "monto",
                                                        "segmento_tx", *OPTIONAL]].to_dict("records")]
            casos.append(CasoSegmentado(cliente_id=cid, segmento=seg, n_tx=len(g),
                                        monto_total=round(float(g["monto"].sum()), 2), transacciones=txs))
        casos.sort(key=lambda c: (-SEG_ORDER[c.segmento], -c.monto_total))
        return IngestaOutput(id_alerta=str(df["id_alerta"].iloc[0]), corte=corte, n_transacciones=len(df),
                             pii_removida=pii, casos=casos)


def segmentar_tx(df: pd.DataFrame) -> pd.Series:
    s = config.rules()["segmentacion"]
    alto = (df["monto"] >= s["alto"]["monto_tx_min"]) | df["producto"].isin(s["alto"]["productos"])
    medio = (df["monto"] >= s["medio"]["monto_tx_min"]) | df["tipo_tarjeta"].isin(s["medio"]["tipos_tarjeta"])
    return pd.Series(["Alto" if a else "Medio" if m else "Bajo" for a, m in zip(alto, medio)], index=df.index)


def _escalar_por_volumen(seg: str, n: int) -> str:
    s = config.rules()["segmentacion"]
    if n >= s["alto"]["n_tx_min"]:
        return "Alto"
    if n >= s["medio"]["n_tx_min"] and seg == "Bajo":
        return "Medio"
    return seg


def _sha(v) -> str:
    return hashlib.sha256(f"{config.HASH_SALT}|{v}".encode()).hexdigest()[:16]


def _clean(r: dict) -> dict:
    return {k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in r.items()}
