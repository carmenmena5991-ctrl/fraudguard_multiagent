# Decisión: cómo reportar el KPI de falsos positivos filtrados

- **Fecha:** 20 de noviembre de 2026
- **Estado:** Aprobada
- **Decisión:** Opción C. Se replantea la meta de la propuesta (> 85 % de FP filtrados) en el informe
  final. El KPI se reporta **segmentado por historial del cliente**, junto con la curva de recall
  contra filtrado y la reducción de carga del analista. No se cambia la política de recall 100 %.

## Contexto

La propuesta fija dos metas que, en IEEE-CIS, no se pueden cumplir a la vez:

| KPI | Meta | Resultado (prueba ciega W18–W22) |
|---|---|---|
| Recall de fraude | 100 % | 100 % |
| FP filtrados automáticamente | > 85 % | 19.3 % |

Prueba ciega: 73 reportes de corte, 2,947 casos, 5.2 % de fraude, modo determinista (sin LLM),
`modelo_riesgo.umbral_bajo = 0.004`.

## Diagnóstico

### La brecha está entera en los clientes sin historial

| Grupo | Legítimos | Filtrados | % filtrado | Fraudes | Recall |
|---|---|---|---|---|---|
| Con historial | 373 | 321 | **86.1 %** | 15 | 100 % |
| Sin historial | 2,422 | 219 | **9.0 %** | 137 | 100 % |
| **Total** | 2,795 | 540 | 19.3 % | 152 | 100 % |

- **Con clientes con historial, la meta de más de 85 % ya se cumple** a recall 100 %.
- Los clientes sin historial son el 87 % de los legítimos y el 90 % de los fraudes.
- De los 2,255 legítimos sin filtrar, 1,517 tienen score de reglas bajo y ninguna regla dura. Lo
  único que los frena es la compuerta "sin historial + score del modelo ≤ `umbral_bajo`".

### El modelo no separa lo suficiente a nivel caso

- AUC 0.93 por transacción (validación temporal), pero **0.81 por caso** (score máximo del caso).
- El fraude sin regla dura con el score más bajo está en 0.0046; la mediana de los legítimos es
  0.015. Las distribuciones se traslapan demasiado.

Curva de recall contra filtrado (prueba ciega):

| umbral_bajo | Recall | FP filtrados | Fraudes omitidos |
|---|---|---|---|
| 0.000 (propuesta literal) | 100 % | 11.5 % | 0 |
| **0.004 (actual)** | **100 %** | **19.3 %** | **0** |
| 0.006 | 98.7 % | 24.5 % | 2 |
| 0.010 | 96.1 % | 34.5 % | 6 |
| 0.020 | 87.5 % | 48.7 % | 19 |
| 0.030 | 82.2 % | 56.3 % | 27 |
| 0.050 | 78.3 % | 62.6 % | 33 |

Ni con 33 fraudes omitidos se llega a 85 %.

## Opciones evaluadas

**A. Activar el LLM para la zona gris: descartada como forma de filtrar.**
`aplicar_politica` (`src/fraudguard/agents/clasificador.py`) convierte en "Requiere Seguimiento
Menor" cualquier "Falso Positivo" del LLM sobre un caso sin historial, salvo que su score sea
≤ `umbral_bajo`, y esos casos ya se descartan por reglas. El LLM solo podría actuar sobre los 52
legítimos con historial que hoy se escalan, así que el filtrado total quedaría en unos 21 % como
máximo. Aunque se quitara esa protección, el LLM ve las mismas métricas que el modelo y no tiene
información nueva para separar mejor.

**B. Mejorar el modelo de riesgo: queda como mejora, no como solución.**
Variables por cliente y por tarjeta pueden subir algunos puntos de AUC por caso, pero la curva
muestra que 85 % exigiría una separación que el dataset no da.

**C. Replantear la meta: elegida.**

## Qué se reporta en el informe final

1. **KPI segmentado.** Con historial: 86.1 % filtrado y 100 % de recall, así que cumple la meta.
   Sin historial: política conservadora a propósito, porque la propuesta prioriza no omitir fraudes.
   Un cliente sin historial solo se descarta si el modelo le da un riesgo casi nulo (≤ 0.004).
2. **Curva de recall contra filtrado** como decisión de negocio. El punto lo escoge el área de
   Prevención de Fraude según el costo de un fraude omitido frente al costo de revisar un caso. Se
   recomienda 0.004 (0 fraudes omitidos).
3. **Reducción de carga del analista.** Los casos escalados llegan auditados, con métricas,
   hallazgos y prioridad. Se respalda con el KPI de tiempo semanal del analista (< 45 min), que
   queda como siguiente medición.
4. **Limitación del dataset.** IEEE-CIS no trae un identificador real de cliente. El cliente se
   aproxima con `card1 + addr1 + (día − D1)`, y por eso 87 % de los casos aparecen sin historial.
   Con datos reales del banco esa proporción debería ser mucho menor y el filtrado total acercarse
   al del grupo con historial.

## Consecuencias

- `scripts/evaluate.py` reporta `casos_por_historial` en `outputs/evaluacion_kpis.json`.
- No cambia `config/rules.json` ni la política de gobernanza.
- El LLM se mantiene para la zona gris (justificación y sugerencia al analista) y para el resumen
  del sintetizador, no como palanca del filtrado.
- Pendientes que se derivan: medir el tiempo semanal del analista y, si hay tiempo, agregar
  variables por cliente al modelo (Opción B) como mejora secundaria.

## Cómo reproducir

```bash
python scripts/evaluate.py --semanas 2018-W18 2018-W19 2018-W20 2018-W21 2018-W22
# outputs/evaluacion_kpis.json → casos, casos_por_historial
```
