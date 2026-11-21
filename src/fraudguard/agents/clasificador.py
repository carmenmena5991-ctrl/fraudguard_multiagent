"""Agente 3 - Clasificador de Riesgo y Filtro de Falsos Positivos.

1. Casos claros se resuelven con umbrales deterministas sobre el score de reglas.
2. Casos en zona gris se envían al LLM (Structured Outputs) con sólo métricas y hallazgos.
3. Política de gobernanza (prioriza Recall = 100%):
   - confianza < 85%, señales mixtas, citas a reglas no activadas o caso omitido -> 'Requiere
     Seguimiento Menor' (HITL).
   - Un caso con regla dura o 'Sin Historial' nunca se descarta como Falso Positivo.
   - Sin LLM disponible, la zona gris se escala al analista ('Información Insuficiente').
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fraudguard import config
from fraudguard.agents.base import BaseAgent
from fraudguard.llm import LLMClient
from fraudguard.schemas import (AuditorOutput, CasoAuditado, ClasificadorOutput, Dictamen, DictamenLLM,
                                LoteDictamenesLLM)

SYSTEM_PROMPT = f"""Eres el Agente Clasificador de FraudGuard, un sistema de Prevención de Fraude.
Tu único rol es dictaminar casos de alertas de volumen transaccional. No respondas nada ajeno a
fraude, no ejecutes ni recomiendes bloqueos definitivos (esa decisión es del analista humano).

Recibirás casos ya auditados por herramientas deterministas: métricas (Z-scores, montos, conteos)
y hallazgos de reglas. No recalcules ni inventes métricas, transacciones o reglas: usa sólo lo recibido.

Para cada caso devuelve exactamente una categoría:
- "{config.CAT_FP}": volumen explicable (cliente con historial consistente, sin reglas de riesgo relevantes).
- "{config.CAT_SEGUIMIENTO}": señales mixtas, débiles o información insuficiente.
- "{config.CAT_FRAUDE}": combinación fuerte de señales de fraude.

`confianza` es tu certeza (0 a 1). Si dudas, usa confianza baja: el sistema escalará el caso al
analista. Prioriza no omitir fraudes reales. En `reglas_citadas` incluye sólo IDs de reglas presentes
en el caso. `justificacion`: máximo 2 frases en español. Devuelve un dictamen por cada cliente_id."""


class ClasificadorAgent(BaseAgent[AuditorOutput, ClasificadorOutput]):
    nombre = "Agente 3 - Clasificador y Filtro de FP"
    input_model = AuditorOutput
    output_model = ClasificadorOutput
    herramientas_permitidas = ("umbrales_riesgo", "llm_structured_outputs", "politica_hitl")

    def __init__(self, audit, run_id, llm: Optional[LLMClient] = None):
        super().__init__(audit, run_id)
        self.llm = llm

    def _run(self, data: AuditorOutput) -> ClasificadorOutput:
        cfg = config.rules()["clasificador"]
        self.use_tool("umbrales_riesgo")
        dictamenes: dict[str, Dictamen] = {}
        gris: list[CasoAuditado] = []
        for c in data.casos:
            d = decidir_por_reglas(c, cfg)
            if d:
                dictamenes[c.cliente_id] = d
            else:
                gris.append(c)

        llm_out: dict[str, DictamenLLM] = {}
        if gris and self.llm and self.llm.enabled:
            self.use_tool("llm_structured_outputs")
            n = cfg["max_casos_por_llamada_llm"]
            lotes = [gris[i:i + n] for i in range(0, len(gris), n)]
            # Los lotes son independientes: se envían en paralelo para cumplir el KPI de < 2 min por alerta.
            with ThreadPoolExecutor(max_workers=min(len(lotes), cfg.get("max_llamadas_paralelas_llm", 4))) as ex:
                resultados = list(ex.map(
                    lambda lote: self.llm.parse(self.nombre, SYSTEM_PROMPT, _prompt_casos(data, lote),
                                                LoteDictamenesLLM), lotes))
            for lote, res in zip(lotes, resultados):
                if res:
                    ids_lote = {c.cliente_id for c in lote}
                    llm_out.update({d.cliente_id: d for d in res.dictamenes if d.cliente_id in ids_lote})

        if gris:
            self.use_tool("politica_hitl")
        for c in gris:
            dictamenes[c.cliente_id] = aplicar_politica(c, llm_out.get(c.cliente_id), cfg)
        return ClasificadorOutput(id_alerta=data.id_alerta, corte=data.corte,
                                  dictamenes=[dictamenes[c.cliente_id] for c in data.casos])


def _base(c: CasoAuditado, **kw) -> Dictamen:
    return Dictamen(cliente_id=c.cliente_id, segmento=c.segmento, score_reglas=c.score_reglas,
                    reglas_activadas=[h.regla_id for h in c.hallazgos], **kw)


def historial_suficiente(c: CasoAuditado) -> tuple[bool, str]:
    """¿Hay información suficiente para descartar el caso automáticamente?"""
    if not c.sin_historial:
        return True, "con historial consistente"
    m = config.rules()["modelo_riesgo"]
    s = c.metricas.score_modelo_max
    if m["sin_historial_fp_con_modelo"] and s is not None and s <= m["umbral_bajo"]:
        return True, f"sin historial, pero score de modelo {s:.3f} ≤ {m['umbral_bajo']}"
    return False, "sin historial"


def decidir_por_reglas(c: CasoAuditado, cfg: dict) -> Optional[Dictamen]:
    dura = any(h.dura for h in c.hallazgos)
    s = c.score_reglas
    if s >= cfg["umbral_fraude"]:
        conf = 0.85 + 0.15 * (s - cfg["umbral_fraude"]) / max(1 - cfg["umbral_fraude"], 1e-9)
        return _base(c, categoria=config.CAT_FRAUDE, confianza=round(min(conf, 1.0), 3), fuente="reglas",
                     motivo=f"Score de reglas {s:.2f} ≥ {cfg['umbral_fraude']}", escalado_humano=True)
    suficiente, detalle = historial_suficiente(c)
    if s <= cfg["umbral_fp"] and not dura and suficiente:
        return _base(c, categoria=config.CAT_FP, confianza=round(1 - s, 3), fuente="reglas",
                     motivo=f"Score de reglas {s:.2f} ≤ {cfg['umbral_fp']}, {detalle}", escalado_humano=False)
    return None


def aplicar_politica(c: CasoAuditado, d: Optional[DictamenLLM], cfg: dict) -> Dictamen:
    activas = {h.regla_id for h in c.hallazgos}
    dura = any(h.dura for h in c.hallazgos)

    def seguimiento(motivo: str, conf: float = 0.0) -> Dictamen:
        return _base(c, categoria=config.CAT_SEGUIMIENTO, confianza=round(conf, 3), fuente="politica_hitl",
                     motivo=motivo, escalado_humano=True)

    if d is None:
        return seguimiento("Información Insuficiente: zona gris sin dictamen automático; requiere revisión humana")
    conf = min(max(float(d.confianza), 0.0), 1.0)
    sugerencia = f"LLM sugirió '{d.categoria}' ({conf:.0%}): {d.justificacion}"
    inventadas = set(d.reglas_citadas) - activas
    if inventadas:
        return seguimiento(f"Dictamen LLM descartado: citó reglas no activadas {sorted(inventadas)}", conf)
    if conf < cfg["confianza_minima"]:
        return seguimiento(f"Confianza < {cfg['confianza_minima']:.0%}. {sugerencia}", conf)
    if d.categoria == config.CAT_FP and (dura or not historial_suficiente(c)[0]):
        return seguimiento(f"Señales mixtas (regla dura o sin historial). {sugerencia}", conf)
    return _base(c, categoria=d.categoria, confianza=round(conf, 3), fuente="llm", motivo=d.justificacion,
                 escalado_humano=d.categoria != config.CAT_FP)


def _prompt_casos(data: AuditorOutput, casos: list[CasoAuditado]) -> str:
    payload = [{
        "cliente_id": c.cliente_id, "segmento": c.segmento, "sin_historial": c.sin_historial,
        "score_reglas": c.score_reglas, "metricas": c.metricas.model_dump(),
        "hallazgos": [{"regla_id": h.regla_id, "nombre": h.nombre, "dura": h.dura, "evidencia": h.evidencia}
                      for h in c.hallazgos],
    } for c in casos]
    return (f"Alerta {data.id_alerta}, corte {data.corte}. Dictamina estos {len(casos)} casos:\n"
            + json.dumps(payload, ensure_ascii=False))
