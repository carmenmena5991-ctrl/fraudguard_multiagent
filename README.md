# FraudGuard MultiAgent

**Automatización de triaje y validación de alertas de volumen transaccional** — Universidad Galileo,
Postgrado en Análisis y Predicción de Datos · Data Science at Work (2026).

Sistema de 4 agentes con gobernanza determinista y supervisión humana (Human-in-the-Loop) que
revisa los reportes de alertas de los cortes **09:00, 12:00 y 16:00**, dictamina cada caso en
*Sin Anomalía (Falso Positivo)*, *Requiere Seguimiento Menor* o *Posible Fraude Confirmado*,
y genera un informe ejecutivo con evidencia trazable.

## Arquitectura

```
 CSV del corte ──► [1] Ingesta y Segmentación ──► [2] Auditor de Reglas e Historial ──►
                    Pandas · Pydantic · SHA-256     Pandas · Z-score · listas de bloqueo ·
                    (rechaza archivos inválidos)    modelo de riesgo (paralelo por segmento)

 ──► [3] Clasificador y Filtro de FP ──► [4] Sintetizador Ejecutivo ──► Streamlit (HITL)
      umbrales + Claude (Structured Outputs)  plantilla Markdown + resumen LLM validado
      política: confianza < 85% → humano

                 Bitácora SQLite transversal (pasos, payloads, herramientas, latencias,
                 llamadas al LLM, decisiones del analista)
```

| Requerimiento | Implementación |
|---|---|
| 1. Multiagente | `src/fraudguard/agents/` — 4 agentes con contrato de entrada/salida propio |
| 2. Gobernanza | Esquemas Pydantic (`schemas.py`, `extra="forbid"`), herramientas permitidas por agente (`BaseAgent.use_tool`), prompts acotados, ningún agente ejecuta bloqueos |
| 3. Sin alucinaciones | Métricas, Z-scores, listas y modelo de riesgo 100% deterministas; el LLM sólo ve zona gris; se descartan dictámenes que citan reglas no activadas y resúmenes que mencionan IDs inexistentes; confianza < 85% → *Seguimiento* |
| 4. Bitácora | `audit.py` → `logs/bitacora.sqlite` (ejecuciones, pasos_agente, llamadas_llm, decisiones_analista), descargable en JSON |
| 5. Interfaz | `app/streamlit_app.py` — carga del corte, progreso por agente, informe, aprobación/corrección, bitácora |
| 6. Construcción propia | Toda la lógica en Python; el LLM (Claude) sólo para razonamiento contextual y síntesis |

## Datos: IEEE-CIS Fraud Detection → reportes de corte

`python -m fraudguard.data_prep` adapta el dataset de Kaggle al caso de uso:

- **Tiempo**: `TransactionDT` se ancla a 2017-12-01. Días < 120 = **Base Histórica**; días ≥ 120 = periodo simulado (9 semanas).
- **Cliente**: `card1 + addr1 + (día − D1)` (aproxima a un cliente real), anonimizado con SHA-256. Tarjeta y dispositivo también se hashean.
- **Reportes de corte**: lunes a viernes, ventanas 16:00 (día anterior)→09:00, 09:00→12:00 y 12:00→16:00. Un cliente entra al reporte si en la ventana tiene ≥ 3 tx o ≥ 1,000 de monto (simula la plataforma de monitoreo). Resultado: 131 reportes, ~23 casos por reporte.
- **Base Histórica**: perfiles por cliente (media/desv. de tx diarias y montos), listas de bloqueo (clientes con fraude, grupos de tarjeta y dispositivos con tasa de fraude ≥ 50%, dominios de email de alto riesgo) y un **modelo de riesgo** (HistGradientBoosting, AUC 0.93 en validación temporal) entrenado sólo con el histórico.
- **Ground truth**: `isFraud` se guarda aparte (`data/ground_truth/`) y nunca llega a los agentes; sólo lo usa `scripts/evaluate.py`.

> Los datos no se versionan (licencia de Kaggle y tamaño). Cada integrante debe descargarlos.

## Instalación

```bash
git clone https://github.com/<usuario>/fraudguard_multiagent.git
cd fraudguard_multiagent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # opcional: ANTHROPIC_API_KEY para habilitar el LLM

# Descargar train_transaction.csv.zip y train_identity.csv.zip de
# https://www.kaggle.com/competitions/ieee-fraud-detection/data
python scripts/extract_data.py ~/Downloads/train_transaction.csv.zip ~/Downloads/train_identity.csv.zip
PYTHONPATH=src python -m fraudguard.data_prep    # ~3 min, ~4 GB RAM
```

## Uso

```bash
streamlit run app/streamlit_app.py                                     # interfaz del analista
PYTHONPATH=src python -m fraudguard.orchestrator data/reports/alerta_20180402_0900.csv --no-llm
python scripts/evaluate.py --semanas 2018-W18 2018-W19 2018-W20 2018-W21 2018-W22   # KPIs
python -m pytest                                                      # tests
```

## Resultados actuales (modo determinista, sin LLM)

Calibración en semanas W14–W17; prueba ciega en W18–W22 (73 reportes, 2,947 casos, prevalencia de fraude 5.2%):

| KPI | Meta | Resultado |
|---|---|---|
| Recall de fraude (casos fraudulentos no descartados) | 100% | **100%** |
| Precisión de *Posible Fraude Confirmado* | — | **100%** (6/6) |
| FP filtrados automáticamente | > 85% | **19.3%** |
| Tiempo de dictamen por alerta | < 2 min | **~0.6 s** (máx. 1.9 s) |

**Trade-off recall vs. filtrado de FP** (umbral del modelo para descartar clientes *Sin Historial*, `config/rules.json → modelo_riesgo.umbral_bajo`):

| umbral_bajo | Recall (prueba) | FP filtrados (prueba) |
|---|---|---|
| 0.000 (propuesta literal: todo cliente sin historial va a revisión) | 100% | 11.5% |
| **0.004 (actual)** | **100%** | **19.3%** |
| 0.010 | 96.1% | 34.5% |
| 0.030 | 82.2% | 56.3% |

El 83% de los casos son clientes sin historial previo, y en IEEE-CIS las reglas de volumen casi no separan fraude: la señal fuerte viene de las listas de bloqueo y del modelo. Las palancas pendientes para acercarse al 85% de filtrado son la zona gris con el LLM habilitado y un mejor modelo de riesgo (features de agregación por cliente).

## Estructura

```
config/rules.json          umbrales, pesos de reglas, segmentación (calibrables sin tocar código)
src/fraudguard/
  config.py · schemas.py · audit.py · llm.py · risk_model.py · data_prep.py · orchestrator.py
  agents/ base.py · ingesta.py · auditor.py · clasificador.py · sintetizador.py
app/streamlit_app.py       interfaz web
scripts/                   extract_data.py · evaluate.py
tests/                     tests de gobernanza y agentes (datos sintéticos)
```

## Cronograma

Construcción: 9 nov – 6 dic 2026 (4 semanas).

## Declaración de uso de IA

El código base de este repositorio se desarrolló con apoyo de Claude Code (Anthropic). La
definición de la tarea, reglas de negocio y tiempos de línea base provienen de la experiencia
laboral de la dueña de la tarea.
