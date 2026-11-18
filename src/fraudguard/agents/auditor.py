"""Agente 2 - Auditor de Reglas e Historial.

Cruza cada caso contra listas de bloqueo y reglas vigentes, y calcula desviaciones (Z-score)
frente a la Base Histórica. 100% determinista (Pandas): el LLM nunca calcula métricas.
Los segmentos Alto / Medio / Bajo se auditan en paralelo.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import pandas as pd

from fraudguard import config, risk_model
from fraudguard.agents.base import AgentError, BaseAgent
from fraudguard.schemas import AuditorOutput, CasoAuditado, CasoSegmentado, Hallazgo, IngestaOutput, MetricasCaso


@lru_cache(maxsize=1)
def base_historica() -> tuple[pd.DataFrame, dict]:
    perfiles_path = config.PROCESSED_DIR / "perfiles_clientes.parquet"
    listas_path = config.PROCESSED_DIR / "listas_bloqueo.json"
    if not perfiles_path.exists() or not listas_path.exists():
        raise AgentError("No existe la Base Histórica. Ejecute: python -m fraudguard.data_prep")
    perfiles = pd.read_parquet(perfiles_path).set_index("cliente_id")
    with open(listas_path, encoding="utf-8") as f:
        listas = json.load(f)
    for k in ("clientes", "tarjetas", "dispositivos", "dominios_email_riesgo"):
        listas[k] = set(listas[k])
    return perfiles, listas


class AuditorAgent(BaseAgent[IngestaOutput, AuditorOutput]):
    nombre = "Agente 2 - Auditor de Reglas e Historial"
    input_model = IngestaOutput
    output_model = AuditorOutput
    herramientas_permitidas = ("consulta_base_historica", "consulta_listas_bloqueo", "motor_reglas", "zscore",
                               "modelo_riesgo")

    def __init__(self, audit, run_id, perfiles: Optional[pd.DataFrame] = None, listas: Optional[dict] = None,
                 usar_modelo: Optional[bool] = None):
        super().__init__(audit, run_id)
        self._perfiles, self._listas = perfiles, listas
        self._usar_modelo = risk_model.disponible() if usar_modelo is None else usar_modelo

    def _run(self, data: IngestaOutput) -> AuditorOutput:
        self.use_tool("consulta_base_historica")
        self.use_tool("consulta_listas_bloqueo")
        if self._perfiles is None:
            self._perfiles, self._listas = base_historica()
        self.use_tool("motor_reglas")
        self.use_tool("zscore")
        if self._usar_modelo:
            self.use_tool("modelo_riesgo")
            # Una sola inferencia vectorizada para todo el reporte (antes de auditar en paralelo)
            ids = [t.transaction_id for c in data.casos for t in c.transacciones]
            self._scores = risk_model.score_transacciones(ids)
        por_segmento: dict[str, list[CasoSegmentado]] = {}
        for c in data.casos:
            por_segmento.setdefault(c.segmento, []).append(c)
        with ThreadPoolExecutor(max_workers=3) as ex:
            resultados = list(ex.map(lambda casos: [self.auditar(c) for c in casos], por_segmento.values()))
        auditados = {c.cliente_id: c for lote in resultados for c in lote}
        casos = [auditados[c.cliente_id] for c in data.casos]  # conserva el orden de ingesta
        return AuditorOutput(id_alerta=data.id_alerta, corte=data.corte, casos=casos)

    def auditar(self, caso: CasoSegmentado) -> CasoAuditado:
        cfg, reglas, listas = config.rules()["auditor"], config.rules()["reglas"], self._listas
        tx = pd.DataFrame([t.model_dump() for t in caso.transacciones])
        perfil = self._perfiles.loc[caso.cliente_id] if caso.cliente_id in self._perfiles.index else None
        dias_hist = int(perfil["dias_activos"]) if perfil is not None else 0
        sin_historial = dias_hist < cfg["min_dias_historial"]

        monto_max = float(tx["monto"].max())
        z_tx = z_monto = ratio = None
        if perfil is not None:
            z_tx = (len(tx) - perfil["tx_diarias_media"]) / max(perfil["tx_diarias_std"], 1.0)
            z_monto = (monto_max - perfil["monto_medio"]) / max(perfil["monto_std"], 1.0)
            ratio = monto_max / max(perfil["monto_medio"], 1.0)
        tarjetas = tx["tarjeta_id"].dropna().unique()
        dispositivos = tx["dispositivo_id"].dropna().unique()
        share_riesgo = float(tx["producto"].isin(cfg["productos_riesgo"]).mean())

        H: list[Hallazgo] = []

        def add(rid: str, evidencia: str, ids=None):
            r = reglas[rid]
            H.append(Hallazgo(regla_id=rid, nombre=r["nombre"], dura=r["dura"], peso=r["peso"],
                              evidencia=evidencia, transaction_ids=[int(i) for i in (ids if ids is not None else [])]))

        if caso.cliente_id in listas["clientes"]:
            add("R01", "Cliente con fraude confirmado en la Base Histórica", tx["transaction_id"])
        hit = tx[tx["tarjeta_id"].isin(listas["tarjetas"])]
        if len(hit):
            add("R02", f"{len(hit)} tx con tarjeta en lista de bloqueo", hit["transaction_id"])
        hit = tx[tx["dispositivo_id"].isin(listas["dispositivos"])]
        if len(hit):
            add("R03", f"{len(hit)} tx desde dispositivo en lista de bloqueo", hit["transaction_id"])
        if z_tx is not None and not sin_historial and z_tx >= cfg["z_count_umbral"]:
            add("R04", f"{len(tx)} tx en el corte vs. media diaria {perfil['tx_diarias_media']:.1f} (Z={z_tx:.1f})")
        if z_monto is not None and not sin_historial and (
                z_monto >= cfg["z_monto_umbral"] or ratio >= cfg["ratio_monto_umbral"]):
            top = tx.loc[tx["monto"].idxmax(), "transaction_id"]
            add("R05", f"Monto máx {monto_max:,.2f} vs. media {perfil['monto_medio']:,.2f} "
                       f"(Z={z_monto:.1f}, x{ratio:.1f})", [top])
        hit = tx[tx["dominio_email"].isin(listas["dominios_email_riesgo"])]
        if len(hit):
            add("R06", f"{len(hit)} tx con dominio de alto riesgo ({', '.join(hit['dominio_email'].unique())})",
                hit["transaction_id"])
        if sin_historial:
            add("R07", f"Cliente con {dias_hist} día(s) en la Base Histórica: 'Sin Historial'")
        if len(tarjetas) >= 2:
            add("R08", f"{len(tarjetas)} tarjetas distintas en el mismo corte")
        if share_riesgo >= cfg["share_producto_riesgo"]:
            add("R09", f"{share_riesgo:.0%} de las tx en producto de alto riesgo {cfg['productos_riesgo']}")
        rafaga = _rafaga(tx, cfg["rafaga_minutos"], cfg["rafaga_n_tx"])
        if rafaga:
            add("R10", f"≥{cfg['rafaga_n_tx']} tx en {cfg['rafaga_minutos']} min", rafaga)

        s_max = s_mean = None
        if self._usar_modelo:
            mcfg = config.rules()["modelo_riesgo"]
            sc = self._scores.reindex(tx["transaction_id"].astype(int)).dropna()
            if len(sc):
                s_max, s_mean = float(sc.max()), float(sc.mean())
                if s_max >= mcfg["umbral_alto"]:
                    hit = sc[sc >= mcfg["umbral_alto"]]
                    add("R11", f"{len(hit)} tx con score de modelo ≥ {mcfg['umbral_alto']} (máx {s_max:.2f})", hit.index)
                elif s_max >= mcfg["umbral_medio"]:
                    hit = sc[sc >= mcfg["umbral_medio"]]
                    add("R12", f"{len(hit)} tx con score de modelo ≥ {mcfg['umbral_medio']} (máx {s_max:.2f})",
                        hit.index)

        score = 1.0
        for h in H:  # noisy-OR: combina pesos sin exceder 1
            score *= (1 - h.peso)
        metricas = MetricasCaso(
            tx_en_corte=len(tx), monto_total=round(float(tx["monto"].sum()), 2), monto_max=round(monto_max, 2),
            z_tx=_r(z_tx), z_monto_max=_r(z_monto), ratio_monto_vs_media=_r(ratio), dias_historial=dias_hist,
            n_tarjetas=len(tarjetas), n_dispositivos=len(dispositivos), share_producto_riesgo=round(share_riesgo, 3),
            score_modelo_max=_r(s_max, 4), score_modelo_medio=_r(s_mean, 4),
        )
        return CasoAuditado(cliente_id=caso.cliente_id, segmento=caso.segmento, sin_historial=sin_historial,
                            metricas=metricas, hallazgos=H, score_reglas=round(1 - score, 4))


def _rafaga(tx: pd.DataFrame, minutos: int, n: int) -> list[int]:
    if len(tx) < n:
        return []
    t = tx.sort_values("fecha_hora")
    ts = pd.to_datetime(t["fecha_hora"]).reset_index(drop=True)
    ids = t["transaction_id"].reset_index(drop=True)
    for i in range(len(ts) - n + 1):
        if ts[i + n - 1] - ts[i] <= pd.Timedelta(minutes=minutos):
            j = i + n - 1
            while j + 1 < len(ts) and ts[j + 1] - ts[i] <= pd.Timedelta(minutes=minutos):
                j += 1
            return ids[i:j + 1].astype(int).tolist()
    return []


def _r(v, nd: int = 2) -> Optional[float]:
    return None if v is None else round(float(v), nd)
