"""Configuración central: rutas, umbrales y parámetros del LLM.

Los valores de negocio (umbrales, pesos de reglas) viven en config/rules.json para que
el equipo los calibre sin tocar código.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"          # reportes de corte simulados (entradas del sistema)
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"  # etiquetas isFraud, nunca se envían a los agentes
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"
RULES_PATH = ROOT / "config" / "rules.json"

AUDIT_DB = Path(os.getenv("FRAUDGUARD_AUDIT_DB", LOGS_DIR / "bitacora.sqlite"))

# --- LLM (Claude) ---
LLM_MODE = os.getenv("FRAUDGUARD_LLM", "auto").lower()   # auto | on | off
LLM_MODEL = os.getenv("FRAUDGUARD_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("FRAUDGUARD_EFFORT", "medium")    # low | medium | high | xhigh | max
LLM_FALLBACKS = os.getenv("FRAUDGUARD_FALLBACKS", "on").lower() == "on"

# Salt para el hashing SHA-256 de identificadores (anonimización).
HASH_SALT = os.getenv("FRAUDGUARD_HASH_SALT", "fraudguard-ieee-cis")

CORTES = ("09:00", "12:00", "16:00")

CATEGORIAS = (
    "Sin Anomalía (Falso Positivo)",
    "Requiere Seguimiento Menor",
    "Posible Fraude Confirmado",
)
CAT_FP, CAT_SEGUIMIENTO, CAT_FRAUDE = CATEGORIAS


@lru_cache(maxsize=1)
def rules() -> dict:
    with open(RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


def llm_enabled() -> bool:
    if LLM_MODE == "off":
        return False
    if LLM_MODE == "on":
        return True
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    return _perfil_ant()


def _perfil_ant() -> bool:
    """¿Hay una credencial guardada por `ant auth login`? El SDK de Anthropic la usa sola."""
    base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "anthropic" / "credentials"
    return base.is_dir() and any(base.glob("*.json"))
