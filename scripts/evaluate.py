"""Evaluación de KPIs contra el ground truth (isFraud) de IEEE-CIS.

Corre el pipeline sobre los reportes de corte de las semanas seleccionadas y calcula:
- Recall de fraude: casos con fraude real que NO fueron descartados como Falso Positivo (meta 100%).
- % de FP filtrados automáticamente: casos legítimos dictaminados 'Sin Anomalía' (meta > 85%).
- Matriz de confusión (categoría del sistema vs. etiqueta real), a nivel caso y a nivel reporte.
- Latencia por alerta (meta < 2 min).

Uso:
    python scripts/evaluate.py                     # todas las semanas, sin LLM
    python scripts/evaluate.py --semanas 2018-W14 2018-W15 --llm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudguard import config  # noqa: E402
from fraudguard.audit import AuditLog  # noqa: E402
from fraudguard.orchestrator import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semanas", nargs="*", help="Semanas ISO (p.ej. 2018-W14). Por defecto, todas.")
    ap.add_argument("--llm", action="store_true", help="Habilita el LLM (consume API)")
    ap.add_argument("--max-reportes", type=int, default=None)
    a = ap.parse_args()

    idx = pd.read_csv(config.REPORTS_DIR / "_indice_reportes.csv")
    if a.semanas:
        idx = idx[idx["semana_iso"].isin(a.semanas)]
    if a.max_reportes:
        idx = idx.head(a.max_reportes)
    gt = pd.read_parquet(config.GROUND_TRUTH_DIR / "etiquetas_transacciones.parquet")
    audit = AuditLog(config.LOGS_DIR / "bitacora_evaluacion.sqlite")

    filas, reportes = [], []
    for i, row in enumerate(idx.itertuples(), 1):
        path = config.REPORTS_DIR / row.archivo
        r = run_pipeline(row.archivo, path.read_bytes(), use_llm=a.llm, audit=audit)
        rep_tx = pd.read_csv(path, usecols=["transaction_id", "cliente_id"]).merge(gt[["transaction_id", "isFraud"]])
        fraude_caso = rep_tx.groupby("cliente_id")["isFraud"].max()
        for d in r.clasificacion.dictamenes:
            caso = next(c for c in r.auditoria.casos if c.cliente_id == d.cliente_id)
            filas.append({"archivo": row.archivo, "semana": row.semana_iso, "cliente_id": d.cliente_id,
                          "categoria": d.categoria, "fuente": d.fuente, "score": d.score_reglas,
                          "score_modelo_max": caso.metricas.score_modelo_max, "sin_historial": caso.sin_historial,
                          "reglas": ",".join(d.reglas_activadas),
                          "sugerencia_llm": d.sugerencia_llm, "confianza_llm": d.confianza_llm,
                          "fraude_real": int(fraude_caso.get(d.cliente_id, 0))})
        reportes.append({"archivo": row.archivo, "semana": row.semana_iso, "latencia_s": r.latencia_s,
                         "dictamen": r.informe.dictamen_reporte, "fraude_real": int(fraude_caso.max())})
        if i % 20 == 0 or i == len(idx):
            print(f"{i}/{len(idx)} reportes procesados", flush=True)

    casos, reps = pd.DataFrame(filas), pd.DataFrame(reportes)
    res = {
        "n_reportes": len(reps), "n_casos": len(casos), "llm": a.llm,
        "casos": metricas(casos),
        # KPI segmentado (decisión docs/decision_kpi_falsos_positivos.md): la meta de > 85% de FP
        # filtrados se evalúa sobre clientes con historial; sin historial la política es conservadora.
        "casos_por_historial": {("sin_historial" if sh else "con_historial"): metricas(g)
                                for sh, g in casos.groupby("sin_historial")},
        "reportes": metricas(reps.rename(columns={"dictamen": "categoria"})),
        "latencia_media_s": round(reps["latencia_s"].mean(), 3), "latencia_max_s": round(reps["latencia_s"].max(), 3),
        "fuente_dictamen": casos["fuente"].value_counts().to_dict(),
    }
    if a.llm:
        res["sugerencias_llm"] = sugerencias_llm(casos)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    casos.to_csv(config.OUTPUTS_DIR / "evaluacion_casos.csv", index=False)
    (config.OUTPUTS_DIR / "evaluacion_kpis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def metricas(df: pd.DataFrame) -> dict:
    fr, leg = df[df["fraude_real"] == 1], df[df["fraude_real"] == 0]
    cm = pd.crosstab(df["categoria"], df["fraude_real"]).reindex(index=list(config.CATEGORIAS), columns=[0, 1],
                                                                    fill_value=0)
    return {
        "prevalencia_fraude": round(df["fraude_real"].mean(), 4),
        "recall_fraude": round((fr["categoria"] != config.CAT_FP).mean(), 4) if len(fr) else None,
        "fp_filtrados": round((leg["categoria"] == config.CAT_FP).mean(), 4) if len(leg) else None,
        "precision_posible_fraude": round(
            df.loc[df["categoria"] == config.CAT_FRAUDE, "fraude_real"].mean(), 4)
        if (df["categoria"] == config.CAT_FRAUDE).any() else None,
        "matriz_confusion": {cat: {"legitimo": int(cm.loc[cat, 0]), "fraude": int(cm.loc[cat, 1])}
                             for cat in cm.index},
    }


def sugerencias_llm(df: pd.DataFrame) -> dict:
    """Qué sugirió el LLM en la zona gris frente a la etiqueta real, antes de aplicar la política HITL.

    Sirve para dos preguntas del informe: ¿el LLM prioriza mejor los fraudes reales? y ¿habría
    descartado fraudes si la política no lo impidiera?
    """
    s = df[df["sugerencia_llm"].notna()]
    if s.empty:
        return {"n_casos": 0}
    tabla = pd.crosstab(s["sugerencia_llm"], s["fraude_real"]).reindex(
        index=list(config.CATEGORIAS), columns=[0, 1], fill_value=0)
    conf = s.groupby("fraude_real")["confianza_llm"].mean()
    return {
        "n_casos": len(s),
        "sugerencia_vs_real": {cat: {"legitimo": int(tabla.loc[cat, 0]), "fraude": int(tabla.loc[cat, 1])}
                               for cat in tabla.index},
        "fraudes_que_sugirio_descartar": int(tabla.loc[config.CAT_FP, 1]),
        "confianza_media": {"legitimo": round(float(conf.get(0, float("nan"))), 3),
                            "fraude": round(float(conf.get(1, float("nan"))), 3)},
    }


if __name__ == "__main__":
    main()
