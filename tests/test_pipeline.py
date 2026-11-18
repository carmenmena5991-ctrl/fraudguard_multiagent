"""Tests de gobernanza y agentes con datos sintéticos (no requieren el dataset ni la API)."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from fraudguard import config
from fraudguard.agents import AgentError, AuditorAgent, ClasificadorAgent, IngestaAgent, SintetizadorAgent
from fraudguard.agents.clasificador import aplicar_politica
from fraudguard.audit import AuditLog
from fraudguard.schemas import DictamenLLM, IngestaInput, SintetizadorInput

CLI_OK, CLI_BLOQ, CLI_NUEVO = "a" * 16, "b" * 16, "c" * 16


def _csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()


def _tx(i, cliente, monto=50.0, minuto=0, corte="09:00", tarjeta="d" * 16):
    return {"id_alerta": "ALR-TEST-0900", "corte": corte, "transaction_id": i,
            "fecha_hora": f"2018-04-02 08:{minuto:02d}:00", "cliente_id": cliente, "tarjeta_id": tarjeta,
            "monto": monto, "producto": "W", "tipo_tarjeta": "debit", "red_tarjeta": "visa",
            "dominio_email": "gmail.com", "tipo_dispositivo": None, "dispositivo_id": None}


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "test.sqlite")


@pytest.fixture
def perfiles():
    return pd.DataFrame([
        {"cliente_id": CLI_OK, "dias_activos": 30, "tx_diarias_media": 3.0, "tx_diarias_std": 1.0,
         "tx_total": 90, "monto_medio": 50.0, "monto_std": 10.0, "monto_max": 80.0},
        {"cliente_id": CLI_BLOQ, "dias_activos": 10, "tx_diarias_media": 1.0, "tx_diarias_std": 0.5,
         "tx_total": 10, "monto_medio": 40.0, "monto_std": 5.0, "monto_max": 50.0},
    ]).set_index("cliente_id")


@pytest.fixture
def listas():
    return {"clientes": {CLI_BLOQ}, "tarjetas": set(), "dispositivos": set(), "dominios_email_riesgo": set()}


def _reporte():
    rows = [_tx(1, CLI_OK, 45, 0), _tx(2, CLI_OK, 55, 30), _tx(3, CLI_OK, 50, 59)]
    rows += [_tx(10 + k, CLI_BLOQ, 900, k) for k in range(4)]
    rows += [_tx(20, CLI_NUEVO, 1200, 5)]
    return _csv(rows)


def run_all(audit, perfiles, listas, contenido=None):
    ing = IngestaAgent(audit, "t").run(IngestaInput(nombre_archivo="r.csv", contenido_csv=contenido or _reporte()))
    aud = AuditorAgent(audit, "t", perfiles, listas, usar_modelo=False).run(ing)
    cla = ClasificadorAgent(audit, "t", llm=None).run(aud)
    inf = SintetizadorAgent(audit, "t", llm=None).run(SintetizadorInput(auditoria=aud, clasificacion=cla))
    return ing, aud, cla, inf


# ---------- Agente 1 ----------
def test_rechaza_archivo_no_csv(audit):
    with pytest.raises(AgentError, match="no es un CSV"):
        IngestaAgent(audit, "t").run(IngestaInput(nombre_archivo="foto.png", contenido_csv=b"xx"))


def test_rechaza_columnas_faltantes(audit):
    with pytest.raises(AgentError, match="faltan columnas"):
        IngestaAgent(audit, "t").run(IngestaInput(nombre_archivo="r.csv", contenido_csv=_csv([{"a": 1}])))


def test_rechaza_corte_distinto_al_declarado(audit):
    with pytest.raises(AgentError, match="declaró el corte"):
        IngestaAgent(audit, "t").run(IngestaInput(nombre_archivo="r.csv", corte_declarado="12:00",
                                                  contenido_csv=_reporte()))


def test_anonimiza_pii(audit):
    rows = [dict(_tx(1, "cliente-real-123", 10), nombre="Juan Pérez")]
    out = IngestaAgent(audit, "t").run(IngestaInput(nombre_archivo="r.csv", contenido_csv=_csv(rows)))
    assert "nombre" in out.pii_removida
    assert out.casos[0].cliente_id != "cliente-real-123" and len(out.casos[0].cliente_id) == 16


def test_segmentacion(audit):
    out = IngestaAgent(audit, "t").run(IngestaInput(nombre_archivo="r.csv", contenido_csv=_reporte()))
    seg = {c.cliente_id: c.segmento for c in out.casos}
    assert seg[CLI_BLOQ] == "Alto" and seg[CLI_NUEVO] == "Alto" and seg[CLI_OK] == "Bajo"


# ---------- Agentes 2-4 ----------
def test_pipeline_dictamenes(audit, perfiles, listas):
    _, aud, cla, inf = run_all(audit, perfiles, listas)
    d = {x.cliente_id: x for x in cla.dictamenes}
    assert d[CLI_OK].categoria == config.CAT_FP
    assert d[CLI_BLOQ].categoria != config.CAT_FP          # regla dura nunca se descarta
    assert d[CLI_NUEVO].categoria == config.CAT_SEGUIMIENTO  # sin historial y sin LLM -> HITL
    assert "R01" in d[CLI_BLOQ].reglas_activadas
    assert inf.dictamen_reporte != config.CAT_FP
    assert CLI_BLOQ in inf.markdown


def test_herramienta_no_permitida(audit):
    agente = IngestaAgent(audit, "t")
    with pytest.raises(PermissionError):
        agente.use_tool("llm_structured_outputs")


def test_agente_rechaza_entrada_ajena(audit, perfiles, listas):
    with pytest.raises(TypeError):
        AuditorAgent(audit, "t", perfiles, listas).run(IngestaInput(nombre_archivo="r.csv", contenido_csv=b""))


# ---------- Política HITL del Agente 3 ----------
def _caso_gris(audit, perfiles, listas):
    _, aud, _, _ = run_all(audit, perfiles, listas)
    return next(c for c in aud.casos if c.cliente_id == CLI_BLOQ)


def test_politica_confianza_baja_escala(audit, perfiles, listas):
    c = _caso_gris(audit, perfiles, listas)
    d = DictamenLLM(cliente_id=c.cliente_id, categoria=config.CAT_FRAUDE, confianza=0.6,
                    justificacion="x", reglas_citadas=["R01"])
    out = aplicar_politica(c, d, config.rules()["clasificador"])
    assert out.categoria == config.CAT_SEGUIMIENTO and out.escalado_humano


def test_politica_no_descarta_regla_dura(audit, perfiles, listas):
    c = _caso_gris(audit, perfiles, listas)
    d = DictamenLLM(cliente_id=c.cliente_id, categoria=config.CAT_FP, confianza=0.99,
                    justificacion="x", reglas_citadas=["R01"])
    assert aplicar_politica(c, d, config.rules()["clasificador"]).categoria == config.CAT_SEGUIMIENTO


def test_politica_rechaza_reglas_inventadas(audit, perfiles, listas):
    c = _caso_gris(audit, perfiles, listas)
    d = DictamenLLM(cliente_id=c.cliente_id, categoria=config.CAT_FRAUDE, confianza=0.95,
                    justificacion="x", reglas_citadas=["R99"])
    out = aplicar_politica(c, d, config.rules()["clasificador"])
    assert out.categoria == config.CAT_SEGUIMIENTO and "no activadas" in out.motivo


def test_politica_acepta_llm_confiable(audit, perfiles, listas):
    c = _caso_gris(audit, perfiles, listas)
    d = DictamenLLM(cliente_id=c.cliente_id, categoria=config.CAT_FRAUDE, confianza=0.95,
                    justificacion="Lista de bloqueo y montos altos", reglas_citadas=["R01"])
    out = aplicar_politica(c, d, config.rules()["clasificador"])
    assert out.categoria == config.CAT_FRAUDE and out.fuente == "llm"


# ---------- Bitácora ----------
def test_bitacora_registra_pasos(audit, perfiles, listas):
    run_all(audit, perfiles, listas)
    pasos = audit.table("pasos_agente")
    assert len(pasos) == 4 and set(pasos["estado"]) == {"ok"}
    assert pasos["herramientas"].str.len().gt(2).all()
    audit.log_decision("t", "ALR", CLI_OK, config.CAT_FP, config.CAT_FP, "ok", 12.0)
    assert len(audit.table("decisiones_analista")) == 1
