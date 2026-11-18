"""Cliente LLM (Claude) con Structured Outputs y degradación segura.

El LLM sólo se usa para razonamiento contextual (Agente 3, zona gris) y síntesis del informe
(Agente 4). Cualquier error, rechazo o salida inválida devuelve ``None`` y el agente aplica su
ruta determinista / escalamiento humano. Nunca se le pide al LLM calcular métricas.
"""
from __future__ import annotations

import time
from typing import Optional, TypeVar

from pydantic import BaseModel

from fraudguard import config
from fraudguard.audit import AuditLog, now

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMClient:
    def __init__(self, audit: AuditLog, run_id: str, enabled: Optional[bool] = None):
        self.audit = audit
        self.run_id = run_id
        self.model = config.LLM_MODEL
        self.enabled = config.llm_enabled() if enabled is None else enabled
        self._client = None
        if self.enabled:
            try:
                import anthropic
                self._anthropic = anthropic
                self._client = anthropic.Anthropic(max_retries=2, timeout=120.0)
            except Exception as e:  # SDK ausente o credenciales inválidas
                self.enabled = False
                self.audit.log_llm(run_id, "init", self.model, now(), 0, error=f"init: {e}")

    def parse(self, agente: str, system: str, user: str, schema: type[T], max_tokens: int = 16000) -> Optional[T]:
        if not self.enabled or self._client is None:
            return None
        use_fallbacks = config.LLM_FALLBACKS
        for attempt in range(2):
            inicio, t0 = now(), time.perf_counter()
            kwargs = dict(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": config.LLM_EFFORT},
            )
            if use_fallbacks:
                # Si el modelo declina por política, el API reintenta en un modelo de respaldo.
                kwargs["extra_headers"] = {"anthropic-beta": FALLBACK_BETA}
                kwargs["extra_body"] = {"fallbacks": "default"}
            try:
                resp = self._client.messages.parse(**kwargs)
            except self._anthropic.BadRequestError as e:
                self._log(agente, inicio, t0, error=f"400: {e.message}")
                if use_fallbacks and attempt == 0:
                    use_fallbacks = False  # reintento sin el parámetro beta
                    continue
                return None
            except self._anthropic.RateLimitError as e:
                self._log(agente, inicio, t0, error=f"429 rate limit: {e.message}")
                return None
            except self._anthropic.APIStatusError as e:
                self._log(agente, inicio, t0, error=f"{e.status_code}: {e.message}")
                return None
            except self._anthropic.APIConnectionError as e:
                self._log(agente, inicio, t0, error=f"conexión: {e}")
                return None
            except Exception as e:  # p.ej. salida que no valida contra el esquema
                self._log(agente, inicio, t0, error=f"parse: {e}")
                return None

            usage = resp.usage
            self._log(agente, inicio, t0, usage.input_tokens, usage.output_tokens, resp.stop_reason)
            if resp.stop_reason in ("refusal", "max_tokens"):
                return None
            return resp.parsed_output
        return None

    def _log(self, agente, inicio, t0, in_tok=None, out_tok=None, stop=None, error=None):
        self.audit.log_llm(self.run_id, agente, self.model, inicio, (time.perf_counter() - t0) * 1000,
                           in_tok, out_tok, stop, error)
