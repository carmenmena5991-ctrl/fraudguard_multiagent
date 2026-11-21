"""Tests del camino con LLM usando respuestas simuladas (no llaman a la API ni consumen créditos)."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from fraudguard import config
from fraudguard.agents import ClasificadorAgent, SintetizadorAgent
from fraudguard.llm import LLMClient
from fraudguard.schemas import LoteDictamenesLLM, ResumenLLM, SintetizadorInput

from test_pipeline import CLI_BLOQ, CLI_NUEVO, _csv, _tx, audit, listas, perfiles, run_all  # noqa: F401


class FakeLLM:
    """Sustituye a LLMClient: registra las llamadas y responde con una función dada."""

    enabled = True

    def __init__(self, responder):
        self.responder = responder
        self.llamadas = []

    def parse(self, agente, system, user, schema, max_tokens=16000):
        self.llamadas.append({"agente": agente, "user": user, "schema": schema})
        return self.responder(user, schema)


def _ids_en_prompt(user: str) -> list[str]:
    return [c["cliente_id"] for c in json.loads(user[user.index("["):])]


def _lote(user, categoria, confianza, reglas=("R01",)):
    return LoteDictamenesLLM(dictamenes=[
        {"cliente_id": cid, "categoria": categoria, "confianza": confianza,
         "justificacion": "Dictamen simulado.", "reglas_citadas": list(reglas)}
        for cid in _ids_en_prompt(user)])


def _clasificar(audit, perfiles, listas, llm):
    _, aud, _, _ = run_all(audit, perfiles, listas)
    cla = ClasificadorAgent(audit, "t", llm=llm).run(aud)
    return aud, cla, {d.cliente_id: d for d in cla.dictamenes}


# ---------- Agente 3 con LLM ----------
def test_llm_solo_recibe_zona_gris(audit, perfiles, listas):
    llm = FakeLLM(lambda u, s: _lote(u, config.CAT_SEGUIMIENTO, 0.9, reglas=()))
    aud, cla, d = _clasificar(audit, perfiles, listas, llm)
    enviados = {cid for ll in llm.llamadas for cid in _ids_en_prompt(ll["user"])}
    assert CLI_NUEVO in enviados                      # sin historial -> zona gris
    assert all(d[cid].fuente == "reglas" for cid in set(d) - enviados)
    assert all(ll["schema"] is LoteDictamenesLLM for ll in llm.llamadas)


def test_llm_no_descarta_cliente_sin_historial(audit, perfiles, listas):
    llm = FakeLLM(lambda u, s: _lote(u, config.CAT_FP, 0.99, reglas=()))
    _, _, d = _clasificar(audit, perfiles, listas, llm)
    assert d[CLI_NUEVO].categoria == config.CAT_SEGUIMIENTO
    assert "sin historial" in d[CLI_NUEVO].motivo


def test_llm_fraude_confiable_se_acepta(audit, perfiles, listas):
    def responder(user, schema):
        return LoteDictamenesLLM(dictamenes=[
            {"cliente_id": cid, "categoria": config.CAT_FRAUDE, "confianza": 0.95,
             "justificacion": "Monto alto sin historial.", "reglas_citadas": []}
            for cid in _ids_en_prompt(user)])
    _, _, d = _clasificar(audit, perfiles, listas, FakeLLM(responder))
    assert d[CLI_NUEVO].categoria == config.CAT_FRAUDE
    assert d[CLI_NUEVO].fuente == "llm" and d[CLI_NUEVO].escalado_humano


def test_llm_falla_escala_al_analista(audit, perfiles, listas):
    _, _, d = _clasificar(audit, perfiles, listas, FakeLLM(lambda u, s: None))
    assert d[CLI_NUEVO].categoria == config.CAT_SEGUIMIENTO
    assert d[CLI_NUEVO].motivo.startswith("Información Insuficiente")


def test_llm_omite_un_caso_escala_al_analista(audit, perfiles, listas):
    def responder(user, schema):
        lote = _lote(user, config.CAT_FRAUDE, 0.95, reglas=())
        lote.dictamenes = [x for x in lote.dictamenes if x.cliente_id != CLI_NUEVO]
        return lote
    _, _, d = _clasificar(audit, perfiles, listas, FakeLLM(responder))
    assert d[CLI_NUEVO].categoria == config.CAT_SEGUIMIENTO and d[CLI_NUEVO].fuente == "politica_hitl"


def test_lotes_se_envian_en_paralelo(audit, perfiles, listas, monkeypatch):
    # Tres clientes sin historial -> tres casos en zona gris; con lotes de 1 caso, tres llamadas.
    nuevos = ["1" * 16, "2" * 16, "3" * 16]
    rows = [_tx(100 + i, cid, 1200, i) for i, cid in enumerate(nuevos)]
    _, aud, _, _ = run_all(audit, perfiles, listas, contenido=_csv(rows))
    monkeypatch.setitem(config.rules()["clasificador"], "max_casos_por_llamada_llm", 1)
    # Si los lotes se enviaran uno tras otro, la barrera nunca se completaría (BrokenBarrierError).
    barrera = threading.Barrier(len(nuevos), timeout=5)

    def responder(user, schema):
        barrera.wait()
        return _lote(user, config.CAT_SEGUIMIENTO, 0.9, reglas=())
    llm = FakeLLM(responder)
    cla = ClasificadorAgent(audit, "t", llm=llm).run(aud)
    assert len(llm.llamadas) == len(nuevos)
    assert {d.cliente_id for d in cla.dictamenes} == set(nuevos)
    assert all(d.fuente == "llm" for d in cla.dictamenes)


def test_dictamen_de_otro_lote_se_ignora(audit, perfiles, listas):
    def responder(user, schema):
        lote = _lote(user, config.CAT_FRAUDE, 0.95, reglas=())
        lote.dictamenes.append(lote.dictamenes[0].model_copy(update={"cliente_id": "e" * 16}))
        return lote
    _, _, d = _clasificar(audit, perfiles, listas, FakeLLM(responder))
    assert "e" * 16 not in d


# ---------- Agente 4 con LLM ----------
def _sintetizar(audit, perfiles, listas, resumen):
    _, aud, cla, _ = run_all(audit, perfiles, listas)
    llm = FakeLLM(lambda u, s: resumen)
    return SintetizadorAgent(audit, "t", llm=llm).run(SintetizadorInput(auditoria=aud, clasificacion=cla))


def test_sintetizador_usa_resumen_llm_valido(audit, perfiles, listas):
    r = ResumenLLM(resumen=f"El cliente {CLI_BLOQ} está en lista de bloqueo.",
                   puntos_clave=["Cliente en lista de bloqueo"], recomendacion="Revisar hoy.")
    inf = _sintetizar(audit, perfiles, listas, r)
    assert inf.resumen_fuente == "llm" and "Revisar hoy." in inf.markdown


def test_sintetizador_descarta_ids_inventados(audit, perfiles, listas):
    r = ResumenLLM(resumen=f"El cliente {'f' * 16} es sospechoso.", puntos_clave=[], recomendacion="x")
    inf = _sintetizar(audit, perfiles, listas, r)
    assert inf.resumen_fuente == "plantilla"


def test_sintetizador_sin_respuesta_usa_plantilla(audit, perfiles, listas):
    assert _sintetizar(audit, perfiles, listas, None).resumen_fuente == "plantilla"


# ---------- LLMClient con un SDK simulado ----------
class _BadRequest(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


_ERRORES = SimpleNamespace(BadRequestError=_BadRequest, RateLimitError=type("RL", (Exception,), {}),
                           APIStatusError=type("AS", (Exception,), {}),
                           APIConnectionError=type("AC", (Exception,), {}))


def _cliente(audit, respuestas):
    """LLMClient cuyo SDK devuelve (o lanza) cada elemento de `respuestas` en orden."""
    llm = LLMClient(audit, "t", enabled=False)
    llm.enabled, llm._anthropic = True, _ERRORES
    llamadas = []

    def parse(**kwargs):
        llamadas.append(kwargs)
        r = respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    llm._client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    return llm, llamadas


def _resp(stop_reason="end_turn", parsed="ok"):
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def test_cliente_devuelve_salida_y_registra_tokens(audit):
    llm, llamadas = _cliente(audit, [_resp()])
    assert llm.parse("A3", "sys", "user", ResumenLLM) == "ok"
    assert llamadas[0]["output_format"] is ResumenLLM
    assert llamadas[0]["output_config"] == {"effort": config.LLM_EFFORT}
    fila = audit.table("llamadas_llm").iloc[-1]
    assert fila["input_tokens"] == 100 and fila["stop_reason"] == "end_turn"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_cliente_descarta_rechazo_y_truncado(audit, stop):
    llm, _ = _cliente(audit, [_resp(stop_reason=stop)])
    assert llm.parse("A3", "sys", "user", ResumenLLM) is None


def test_cliente_reintenta_sin_fallbacks_si_400(audit, monkeypatch):
    monkeypatch.setattr(config, "LLM_FALLBACKS", True)
    llm, llamadas = _cliente(audit, [_BadRequest("fallbacks no soportado"), _resp()])
    assert llm.parse("A3", "sys", "user", ResumenLLM) == "ok"
    assert "extra_body" in llamadas[0] and "extra_body" not in llamadas[1]
    assert audit.table("llamadas_llm")["error"].iloc[0].startswith("400")


def test_cliente_error_de_conexion_devuelve_none(audit):
    llm, _ = _cliente(audit, [_ERRORES.APIConnectionError("sin red")])
    assert llm.parse("A3", "sys", "user", ResumenLLM) is None
    assert "conexión" in audit.table("llamadas_llm")["error"].iloc[-1]
