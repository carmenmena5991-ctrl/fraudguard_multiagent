"""Orquestador: Ingesta -> Auditor -> Clasificador -> Sintetizador, con bitácora de cada paso.

Uso por línea de comandos:
    python -m fraudguard.orchestrator data/reports/alerta_20180402_0900.csv [--no-llm]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from fraudguard import config
from fraudguard.agents import AgentError, AuditorAgent, ClasificadorAgent, IngestaAgent, SintetizadorAgent
from fraudguard.audit import AuditLog
from fraudguard.llm import LLMClient
from fraudguard.schemas import (AuditorOutput, ClasificadorOutput, InformeEjecutivo, IngestaInput, IngestaOutput,
                                SintetizadorInput)

Progress = Callable[[int, str, str], None]  # (n_agente, nombre, estado)


@dataclass
class ResultadoPipeline:
    run_id: str
    latencia_s: float
    ingesta: IngestaOutput
    auditoria: AuditorOutput
    clasificacion: ClasificadorOutput
    informe: InformeEjecutivo
    llm_habilitado: bool


def run_pipeline(nombre_archivo: str, contenido: bytes, corte_declarado: Optional[str] = None,
                 use_llm: Optional[bool] = None, audit: Optional[AuditLog] = None,
                 progress: Optional[Progress] = None) -> ResultadoPipeline:
    audit = audit or AuditLog()
    llm_on = config.llm_enabled() if use_llm is None else use_llm
    run_id = audit.start_run(nombre_archivo, llm_on, config.LLM_MODEL if llm_on else "—")
    llm = LLMClient(audit, run_id, enabled=llm_on)
    notify = progress or (lambda *_: None)
    t0 = time.perf_counter()
    paso = (0, "")
    try:
        paso = (1, IngestaAgent.nombre); notify(*paso, "en_proceso")
        ing = IngestaAgent(audit, run_id).run(
            IngestaInput(nombre_archivo=nombre_archivo, corte_declarado=corte_declarado, contenido_csv=contenido))
        notify(*paso, "ok")

        paso = (2, AuditorAgent.nombre); notify(*paso, "en_proceso")
        aud = AuditorAgent(audit, run_id).run(ing)
        notify(*paso, "ok")

        paso = (3, ClasificadorAgent.nombre); notify(*paso, "en_proceso")
        cla = ClasificadorAgent(audit, run_id, llm).run(aud)
        notify(*paso, "ok")

        paso = (4, SintetizadorAgent.nombre); notify(*paso, "en_proceso")
        inf = SintetizadorAgent(audit, run_id, llm).run(SintetizadorInput(auditoria=aud, clasificacion=cla))
        notify(*paso, "ok")
    except Exception as e:
        notify(*paso, "error")
        audit.end_run(run_id, estado="rechazado" if isinstance(e, AgentError) else "error",
                      latencia_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
        raise
    lat = time.perf_counter() - t0
    audit.end_run(run_id, estado="completado", latencia_s=lat, id_alerta=inf.id_alerta, corte=inf.corte,
                  n_casos=len(cla.dictamenes), dictamen_reporte=inf.dictamen_reporte)
    return ResultadoPipeline(run_id, lat, ing, aud, cla, inf, llm.enabled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archivo", type=Path)
    ap.add_argument("--no-llm", action="store_true", help="Modo determinista (sin llamadas al LLM)")
    a = ap.parse_args()
    r = run_pipeline(a.archivo.name, a.archivo.read_bytes(), use_llm=False if a.no_llm else None,
                     progress=lambda n, nombre, est: print(f"[{est:>10}] {nombre}"))
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    out = config.OUTPUTS_DIR / f"informe_{r.informe.id_alerta}.md"
    out.write_text(r.informe.markdown, encoding="utf-8")
    print(f"\n{r.informe.markdown}\n\nrun_id={r.run_id} · {r.latencia_s:.2f}s · LLM={'sí' if r.llm_habilitado else 'no'}"
          f"\nInforme guardado en {out}")


if __name__ == "__main__":
    main()
