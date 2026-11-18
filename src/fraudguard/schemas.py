"""Contratos de entrada/salida de cada agente (gobernanza por esquema).

Cada agente sólo acepta su modelo de entrada y sólo puede emitir su modelo de salida;
el orquestador valida ambos lados.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Categoria = Literal[
    "Sin Anomalía (Falso Positivo)",
    "Requiere Seguimiento Menor",
    "Posible Fraude Confirmado",
]
Segmento = Literal["Alto", "Medio", "Bajo"]
Corte = Literal["09:00", "12:00", "16:00"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Agente 1: Ingesta y Segmentación ----------
class Transaccion(Strict):
    transaction_id: int
    fecha_hora: datetime
    cliente_id: str = Field(min_length=8)
    tarjeta_id: Optional[str] = None
    monto: float = Field(ge=0)
    producto: Optional[str] = None
    tipo_tarjeta: Optional[str] = None
    red_tarjeta: Optional[str] = None
    region: Optional[float] = None
    distancia: Optional[float] = None
    dominio_email: Optional[str] = None
    tipo_dispositivo: Optional[str] = None
    dispositivo_id: Optional[str] = None
    segmento_tx: Segmento


class CasoSegmentado(Strict):
    cliente_id: str
    segmento: Segmento
    n_tx: int
    monto_total: float
    transacciones: list[Transaccion]


class IngestaInput(Strict):
    nombre_archivo: str
    corte_declarado: Optional[Corte] = None
    contenido_csv: bytes


class IngestaOutput(Strict):
    id_alerta: str
    corte: Corte
    n_transacciones: int
    pii_removida: list[str]
    casos: list[CasoSegmentado]


# ---------- Agente 2: Auditor de Reglas e Historial ----------
class Hallazgo(Strict):
    regla_id: str
    nombre: str
    dura: bool
    peso: float
    evidencia: str
    transaction_ids: list[int] = []


class MetricasCaso(Strict):
    tx_en_corte: int
    monto_total: float
    monto_max: float
    z_tx: Optional[float] = None
    z_monto_max: Optional[float] = None
    ratio_monto_vs_media: Optional[float] = None
    dias_historial: int = 0
    n_tarjetas: int
    n_dispositivos: int
    share_producto_riesgo: float
    score_modelo_max: Optional[float] = None
    score_modelo_medio: Optional[float] = None


class CasoAuditado(Strict):
    cliente_id: str
    segmento: Segmento
    sin_historial: bool
    metricas: MetricasCaso
    hallazgos: list[Hallazgo]
    score_reglas: float = Field(ge=0, le=1)


class AuditorOutput(Strict):
    id_alerta: str
    corte: Corte
    casos: list[CasoAuditado]


# ---------- Agente 3: Clasificador y Filtro de FP ----------
class DictamenLLM(BaseModel):
    """Salida estructurada que se le exige al LLM (Structured Outputs)."""
    cliente_id: str
    categoria: Categoria
    confianza: float = Field(description="Confianza entre 0 y 1")
    justificacion: str = Field(description="Máximo 2 frases, citando sólo métricas y reglas recibidas")
    reglas_citadas: list[str]


class LoteDictamenesLLM(BaseModel):
    dictamenes: list[DictamenLLM]


class Dictamen(Strict):
    cliente_id: str
    segmento: Segmento
    categoria: Categoria
    confianza: float = Field(ge=0, le=1)
    fuente: Literal["reglas", "llm", "politica_hitl"]
    motivo: str
    escalado_humano: bool
    score_reglas: float
    reglas_activadas: list[str]


class ClasificadorOutput(Strict):
    id_alerta: str
    corte: Corte
    dictamenes: list[Dictamen]


# ---------- Agente 4: Sintetizador Ejecutivo ----------
class ResumenLLM(BaseModel):
    resumen: str = Field(description="3 a 5 frases, sólo con datos recibidos")
    puntos_clave: list[str]
    recomendacion: str


class SintetizadorInput(Strict):
    auditoria: AuditorOutput
    clasificacion: ClasificadorOutput


class InformeEjecutivo(Strict):
    id_alerta: str
    corte: Corte
    dictamen_reporte: Categoria
    conteo_por_categoria: dict[str, int]
    resumen: str
    resumen_fuente: Literal["llm", "plantilla"]
    markdown: str
