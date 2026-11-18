"""Agente 4 - Sintetizador Ejecutivo.

Construye el informe con evidencia trazable (IDs, montos, reglas violadas) de forma determinista.
El LLM sólo redacta el resumen narrativo; si menciona identificadores que no existen en la
evidencia, el resumen se descarta y se usa la plantilla (anti-alucinación).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from fraudguard import config
from fraudguard.agents.base import BaseAgent
from fraudguard.llm import LLMClient
from fraudguard.schemas import InformeEjecutivo, ResumenLLM, SintetizadorInput

SEVERIDAD = {config.CAT_FP: 0, config.CAT_SEGUIMIENTO: 1, config.CAT_FRAUDE: 2}
HEX_ID = re.compile(r"\b[0-9a-f]{16}\b")
MAX_CASOS_LLM = 30

SYSTEM_PROMPT = """Eres el Agente Sintetizador de FraudGuard (Prevención de Fraude). Redactas en español
un resumen ejecutivo para el analista de fraude a partir EXCLUSIVAMENTE de los datos recibidos.
No inventes clientes, montos, transacciones ni reglas. No recomiendes bloqueos definitivos como
hechos consumados: la decisión final es del analista. Sé breve y concreto."""


class SintetizadorAgent(BaseAgent[SintetizadorInput, InformeEjecutivo]):
    nombre = "Agente 4 - Sintetizador Ejecutivo"
    input_model = SintetizadorInput
    output_model = InformeEjecutivo
    herramientas_permitidas = ("plantilla_markdown", "llm_structured_outputs", "validador_evidencia")

    def __init__(self, audit, run_id, llm: Optional[LLMClient] = None):
        super().__init__(audit, run_id)
        self.llm = llm

    def _run(self, data: SintetizadorInput) -> InformeEjecutivo:
        aud, cla = data.auditoria, data.clasificacion
        casos = {c.cliente_id: c for c in aud.casos}
        conteo = {cat: 0 for cat in config.CATEGORIAS}
        for d in cla.dictamenes:
            conteo[d.categoria] += 1
        dictamen_reporte = max((d.categoria for d in cla.dictamenes), key=SEVERIDAD.get, default=config.CAT_FP)
        sospechosos = sorted([d for d in cla.dictamenes if d.categoria != config.CAT_FP],
                             key=lambda d: (-SEVERIDAD[d.categoria], -d.score_reglas))

        self.use_tool("plantilla_markdown")
        resumen, fuente = _resumen_plantilla(aud.id_alerta, aud.corte, conteo, sospechosos, casos), "plantilla"
        if sospechosos and self.llm and self.llm.enabled:
            self.use_tool("llm_structured_outputs")
            evidencia = [{
                "cliente_id": d.cliente_id, "categoria": d.categoria, "confianza": d.confianza,
                "segmento": d.segmento, "monto_total": casos[d.cliente_id].metricas.monto_total,
                "tx_en_corte": casos[d.cliente_id].metricas.tx_en_corte,
                "reglas": [f"{h.regla_id}: {h.evidencia}" for h in casos[d.cliente_id].hallazgos],
            } for d in sospechosos[:MAX_CASOS_LLM]]
            user = (f"Alerta {aud.id_alerta} (corte {aud.corte}). Conteo por categoría: "
                    f"{json.dumps(conteo, ensure_ascii=False)}. Casos que requieren atención:\n"
                    f"{json.dumps(evidencia, ensure_ascii=False)}")
            r = self.llm.parse(self.nombre, SYSTEM_PROMPT, user, ResumenLLM)
            self.use_tool("validador_evidencia")
            if r and _ids_validos(r, set(casos)):
                puntos = "\n".join(f"- {p}" for p in r.puntos_clave)
                resumen, fuente = f"{r.resumen}\n\n{puntos}\n\n**Recomendación:** {r.recomendacion}", "llm"

        md = _markdown(aud.id_alerta, aud.corte, dictamen_reporte, conteo, resumen, fuente, sospechosos, casos)
        return InformeEjecutivo(id_alerta=aud.id_alerta, corte=aud.corte, dictamen_reporte=dictamen_reporte,
                                conteo_por_categoria=conteo, resumen=resumen, resumen_fuente=fuente, markdown=md)


def _ids_validos(r: ResumenLLM, conocidos: set[str]) -> bool:
    texto = " ".join([r.resumen, r.recomendacion, *r.puntos_clave])
    return set(HEX_ID.findall(texto)) <= conocidos


def _resumen_plantilla(id_alerta, corte, conteo, sospechosos, casos) -> str:
    n_fr, n_seg, n_fp = conteo[config.CAT_FRAUDE], conteo[config.CAT_SEGUIMIENTO], conteo[config.CAT_FP]
    total = n_fr + n_seg + n_fp
    txt = (f"La alerta {id_alerta} (corte {corte}) contiene {total} casos: {n_fr} con posible fraude, "
           f"{n_seg} que requieren seguimiento y {n_fp} descartados como falsos positivos.")
    if sospechosos:
        top = sospechosos[0]
        m = casos[top.cliente_id].metricas
        txt += (f" El caso prioritario es el cliente {top.cliente_id} ({top.categoria}, score {top.score_reglas:.2f}, "
                f"{m.tx_en_corte} tx por {m.monto_total:,.2f}; reglas {', '.join(top.reglas_activadas) or '—'}).")
    return txt


def _markdown(id_alerta, corte, dictamen, conteo, resumen, fuente, sospechosos, casos) -> str:
    lines = [
        f"# Informe Ejecutivo — {id_alerta}",
        f"**Corte:** {corte} · **Dictamen del reporte:** {dictamen}",
        "",
        "| Categoría | Casos |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in conteo.items()],
        "",
        f"## Resumen ({'generado por LLM y validado' if fuente == 'llm' else 'plantilla determinista'})",
        resumen,
        "",
        "## Evidencia de casos que requieren acción",
    ]
    if not sospechosos:
        lines.append("_Sin casos sospechosos: todos los casos se descartaron como falsos positivos._")
    else:
        lines += ["| Cliente | Categoría | Conf. | Segmento | Tx | Monto total | Reglas | Tx IDs (muestra) | Motivo |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for d in sospechosos:
            c = casos[d.cliente_id]
            ids = sorted({i for h in c.hallazgos for i in h.transaction_ids})[:5]
            motivo = d.motivo.replace("|", "/").replace("\n", " ")
            lines.append(f"| `{d.cliente_id}` | {d.categoria} | {d.confianza:.0%} | {d.segmento} | "
                         f"{c.metricas.tx_en_corte} | {c.metricas.monto_total:,.2f} | "
                         f"{', '.join(d.reglas_activadas) or '—'} | {', '.join(map(str, ids)) or '—'} | {motivo} |")
    lines += ["", "> FraudGuard no ejecuta bloqueos: el analista revisa y emite la decisión final."]
    return "\n".join(lines)
