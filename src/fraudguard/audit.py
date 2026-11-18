"""Bitácora auditable transversal (SQLite).

Registra cada ejecución: ID de alerta, timestamps, agente activo, payloads de entrada/salida,
herramientas invocadas, latencia, llamadas al LLM y la decisión final del analista.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraudguard import config

MAX_PAYLOAD_CHARS = 200_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS ejecuciones (
    run_id TEXT PRIMARY KEY, id_alerta TEXT, corte TEXT, archivo TEXT,
    inicio TEXT, fin TEXT, latencia_s REAL, estado TEXT, n_casos INTEGER,
    llm_habilitado INTEGER, modelo TEXT, dictamen_reporte TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS pasos_agente (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, agente TEXT,
    inicio TEXT, fin TEXT, latencia_ms REAL, estado TEXT,
    payload_entrada TEXT, payload_salida TEXT, herramientas TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS llamadas_llm (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, agente TEXT, modelo TEXT,
    inicio TEXT, latencia_ms REAL, input_tokens INTEGER, output_tokens INTEGER,
    stop_reason TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS decisiones_analista (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, id_alerta TEXT, cliente_id TEXT,
    categoria_sistema TEXT, decision_analista TEXT, comentario TEXT,
    decidido_en TEXT, segundos_revision REAL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dump(obj) -> str:
    if obj is None:
        return ""
    if hasattr(obj, "model_dump_json"):
        s = obj.model_dump_json()
    else:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= MAX_PAYLOAD_CHARS else s[:MAX_PAYLOAD_CHARS] + "...[truncado]"


class AuditLog:
    def __init__(self, db_path: Path | str = config.AUDIT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- ejecuciones ---
    def start_run(self, archivo: str, llm_habilitado: bool, modelo: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO ejecuciones (run_id, archivo, inicio, estado, llm_habilitado, modelo) VALUES (?,?,?,?,?,?)",
                (run_id, archivo, now(), "en_proceso", int(llm_habilitado), modelo),
            )
        return run_id

    def end_run(self, run_id: str, *, estado: str, latencia_s: float, id_alerta=None, corte=None,
                n_casos=None, dictamen_reporte=None, error=None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE ejecuciones SET fin=?, estado=?, latencia_s=?, id_alerta=?, corte=?, n_casos=?, "
                "dictamen_reporte=?, error=? WHERE run_id=?",
                (now(), estado, latencia_s, id_alerta, corte, n_casos, dictamen_reporte, error, run_id),
            )

    # --- pasos ---
    def log_step(self, run_id: str, agente: str, inicio: str, latencia_ms: float, estado: str,
                 entrada=None, salida=None, herramientas: list[str] | None = None, error: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO pasos_agente (run_id, agente, inicio, fin, latencia_ms, estado, payload_entrada, "
                "payload_salida, herramientas, error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, agente, inicio, now(), latencia_ms, estado, _dump(entrada), _dump(salida),
                 json.dumps(herramientas or [], ensure_ascii=False), error),
            )

    def log_llm(self, run_id: str, agente: str, modelo: str, inicio: str, latencia_ms: float,
                input_tokens=None, output_tokens=None, stop_reason=None, error=None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO llamadas_llm (run_id, agente, modelo, inicio, latencia_ms, input_tokens, "
                "output_tokens, stop_reason, error) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, agente, modelo, inicio, latencia_ms, input_tokens, output_tokens, stop_reason, error),
            )

    # --- Human-in-the-Loop ---
    def log_decision(self, run_id: str, id_alerta: str, cliente_id: str, categoria_sistema: str,
                     decision_analista: str, comentario: str = "", segundos_revision: float | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO decisiones_analista (run_id, id_alerta, cliente_id, categoria_sistema, "
                "decision_analista, comentario, decidido_en, segundos_revision) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, id_alerta, cliente_id, categoria_sistema, decision_analista, comentario, now(),
                 segundos_revision),
            )

    # --- consulta / exportación ---
    def table(self, name: str, run_id: str | None = None) -> pd.DataFrame:
        if name not in {"ejecuciones", "pasos_agente", "llamadas_llm", "decisiones_analista"}:
            raise ValueError(f"tabla desconocida: {name}")
        q, params = f"SELECT * FROM {name}", ()
        if run_id:
            q, params = q + " WHERE run_id = ?", (run_id,)
        with self._conn() as c:
            return pd.read_sql_query(q, c, params=params)

    def export_json(self, run_id: str | None = None) -> str:
        data = {t: self.table(t, run_id).to_dict(orient="records")
                for t in ("ejecuciones", "pasos_agente", "llamadas_llm", "decisiones_analista")}
        return json.dumps(data, ensure_ascii=False, indent=1, default=str)
