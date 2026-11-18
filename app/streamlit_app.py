"""FraudGuard MultiAgent - Interfaz web del analista (Streamlit).

Ejecutar:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudguard import config  # noqa: E402
from fraudguard.agents import AgentError  # noqa: E402
from fraudguard.audit import AuditLog  # noqa: E402
from fraudguard.orchestrator import run_pipeline  # noqa: E402

st.set_page_config(page_title="FraudGuard MultiAgent", page_icon="🛡️", layout="wide")
audit = AuditLog()

AGENTES = ["Agente 1 - Ingesta y Segmentación", "Agente 2 - Auditor de Reglas e Historial",
           "Agente 3 - Clasificador y Filtro de FP", "Agente 4 - Sintetizador Ejecutivo"]
ICONO = {"en_proceso": "⏳", "ok": "✅", "error": "❌", "pendiente": "⬜"}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🛡️ FraudGuard")
    st.caption("Triaje y validación de alertas de volumen transaccional")
    llm_ok = config.llm_enabled()
    use_llm = st.toggle("Usar LLM (Claude) en zona gris y resumen", value=llm_ok, disabled=not llm_ok,
                        help=None if llm_ok else "Configure ANTHROPIC_API_KEY en .env para habilitarlo")
    st.caption(f"Modelo: `{config.LLM_MODEL}`" if llm_ok else "Modo determinista (sin LLM)")
    corte = st.selectbox("Corte horario", config.CORTES)
    fuente = st.radio("Origen del reporte", ["Reportes simulados (IEEE-CIS)", "Subir archivo"])
    archivo, contenido = None, None
    if fuente == "Subir archivo":
        up = st.file_uploader("Reporte del corte (CSV)", type=None)
        if up:
            archivo, contenido = up.name, up.getvalue()
    else:
        idx_path = config.REPORTS_DIR / "_indice_reportes.csv"
        if idx_path.exists():
            idx = pd.read_csv(idx_path)
            idx = idx[idx["corte"] == corte]
            sel = st.selectbox("Reporte", idx["archivo"].tolist())
            if sel:
                archivo, contenido = sel, (config.REPORTS_DIR / sel).read_bytes()
        else:
            st.warning("No hay reportes. Ejecute `python -m fraudguard.data_prep`.")
    ejecutar = st.button("▶ Ejecutar pipeline", type="primary", disabled=contenido is None, width="stretch")

st.title("Revisión de alerta de volumen transaccional")

# ---------------- Ejecución ----------------
if ejecutar:
    estado = {a: "pendiente" for a in AGENTES}
    box = st.empty()

    def render():
        box.markdown("  \n".join(f"{ICONO[estado[a]]} **{a}**" for a in AGENTES))

    def progress(n, nombre, est):
        estado[nombre] = est
        render()

    render()
    try:
        r = run_pipeline(archivo, contenido, corte_declarado=corte, use_llm=use_llm, audit=audit, progress=progress)
        st.session_state["resultado"] = r
        st.session_state["t_mostrado"] = time.time()
    except AgentError as e:
        st.session_state.pop("resultado", None)
        st.error(f"⛔ Flujo detenido por el Agente de Ingesta: {e}")
    except Exception as e:
        st.session_state.pop("resultado", None)
        st.exception(e)

r = st.session_state.get("resultado")
if r is None:
    st.info("Seleccione el corte y el reporte en la barra lateral, luego ejecute el pipeline.")
    st.stop()

inf, cla, aud = r.informe, r.clasificacion, r.auditoria
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Alerta", inf.id_alerta)
c2.metric("Casos", len(cla.dictamenes))
c3.metric("Posible fraude", inf.conteo_por_categoria[config.CAT_FRAUDE])
c4.metric("Seguimiento", inf.conteo_por_categoria[config.CAT_SEGUIMIENTO])
c5.metric("Tiempo de dictamen", f"{r.latencia_s:.1f} s")

tab_inf, tab_casos, tab_bit = st.tabs(["📄 Informe ejecutivo", "🧑‍⚖️ Revisión de casos (HITL)", "📚 Bitácora"])

with tab_inf:
    st.markdown(inf.markdown)
    st.download_button("Descargar informe (.md)", inf.markdown, file_name=f"informe_{inf.id_alerta}.md")

with tab_casos:
    st.caption("Apruebe o corrija el dictamen. Los casos descartados como FP se muestran al final para auditoría.")
    casos = {c.cliente_id: c for c in aud.casos}
    df = pd.DataFrame([{
        "cliente_id": d.cliente_id, "categoria_sistema": d.categoria, "confianza": d.confianza,
        "fuente": d.fuente, "segmento": d.segmento, "tx": casos[d.cliente_id].metricas.tx_en_corte,
        "monto_total": casos[d.cliente_id].metricas.monto_total,
        "score_modelo": casos[d.cliente_id].metricas.score_modelo_max,
        "reglas": ", ".join(d.reglas_activadas), "motivo": d.motivo,
        "decision_analista": d.categoria, "comentario": "",
    } for d in cla.dictamenes])
    orden = {config.CAT_FRAUDE: 0, config.CAT_SEGUIMIENTO: 1, config.CAT_FP: 2}
    df = df.sort_values("categoria_sistema", key=lambda s: s.map(orden)).reset_index(drop=True)
    editado = st.data_editor(
        df, hide_index=True, width="stretch",
        disabled=[c for c in df.columns if c not in ("decision_analista", "comentario")],
        column_config={
            "decision_analista": st.column_config.SelectboxColumn("Decisión analista", options=list(config.CATEGORIAS),
                                                                  required=True),
            "confianza": st.column_config.ProgressColumn("Confianza", min_value=0, max_value=1, format="%.2f"),
        },
    )
    if st.button("💾 Registrar decisiones del analista", type="primary"):
        seg = time.time() - st.session_state.get("t_mostrado", time.time())
        for row in editado.itertuples():
            audit.log_decision(r.run_id, inf.id_alerta, row.cliente_id, row.categoria_sistema,
                               row.decision_analista, row.comentario or "", seg)
        cambios = int((editado["decision_analista"] != editado["categoria_sistema"]).sum())
        st.success(f"{len(editado)} decisiones registradas ({cambios} correcciones) · revisión: {seg / 60:.1f} min")

    with st.expander("Detalle de hallazgos por caso"):
        cid = st.selectbox("Cliente", df["cliente_id"])
        c = casos[cid]
        st.json(c.model_dump(mode="json"))

with tab_bit:
    st.subheader("Pasos de esta ejecución")
    pasos = audit.table("pasos_agente", r.run_id)
    st.dataframe(pasos[["agente", "inicio", "latencia_ms", "estado", "herramientas", "error"]],
                 hide_index=True, width="stretch")
    llm_calls = audit.table("llamadas_llm", r.run_id)
    if len(llm_calls):
        st.subheader("Llamadas al LLM")
        st.dataframe(llm_calls, hide_index=True, width="stretch")
    st.subheader("Historial de ejecuciones")
    st.dataframe(audit.table("ejecuciones").sort_values("inicio", ascending=False), hide_index=True,
                 width="stretch")
    st.download_button("Descargar bitácora de esta ejecución (JSON)", audit.export_json(r.run_id),
                       file_name=f"bitacora_{r.run_id}.json")
    st.download_button("Descargar bitácora completa (JSON)", audit.export_json(), file_name="bitacora_completa.json")
