"""Clase base: cada agente declara su contrato de entrada/salida y sus herramientas permitidas."""
from __future__ import annotations

import time
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from fraudguard.audit import AuditLog, now

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)


class AgentError(Exception):
    """Error de negocio que detiene el flujo y requiere intervención del analista."""


class BaseAgent(Generic[I, O]):
    nombre: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]
    herramientas_permitidas: ClassVar[tuple[str, ...]] = ()

    def __init__(self, audit: AuditLog, run_id: str):
        self.audit = audit
        self.run_id = run_id
        self._tools_used: list[str] = []

    def use_tool(self, name: str) -> None:
        if name not in self.herramientas_permitidas:
            raise PermissionError(f"{self.nombre} no tiene permitido usar la herramienta '{name}'")
        if name not in self._tools_used:
            self._tools_used.append(name)

    def run(self, data: I) -> O:
        if not isinstance(data, self.input_model):
            raise TypeError(f"{self.nombre} sólo acepta {self.input_model.__name__}")
        inicio, t0 = now(), time.perf_counter()
        self._tools_used = []
        try:
            out = self._run(data)
            out = self.output_model.model_validate(out.model_dump())  # re-validación de salida
        except Exception as e:
            self.audit.log_step(self.run_id, self.nombre, inicio, (time.perf_counter() - t0) * 1000,
                                "error", entrada=self._loggable_input(data), herramientas=self._tools_used,
                                error=f"{type(e).__name__}: {e}")
            raise
        self.audit.log_step(self.run_id, self.nombre, inicio, (time.perf_counter() - t0) * 1000, "ok",
                            entrada=self._loggable_input(data), salida=out, herramientas=self._tools_used)
        return out

    def _loggable_input(self, data: I):
        return data

    def _run(self, data: I) -> O:  # pragma: no cover
        raise NotImplementedError
