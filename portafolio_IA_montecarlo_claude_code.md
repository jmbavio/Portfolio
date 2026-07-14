# Portafolio IA (riesgo + estable) — paquete para Claude Code

## 1. Estructura del portafolio (capital total: $3.000.000 ARS)

### Sleeve riesgo — $1.000.000 ARS (33%)
| Ticker | Rol | Peso ARS | % del sleeve | Ratio CEDEAR (BYMA) |
|---|---|---|---|---|
| SMH | Semis diversificado (chips+equipos) | 350.000 | 35% | 50:1 |
| NVDA | Chip líder, convicción | 150.000 | 15% | 24:1 |
| CEG | Energía nuclear (data centers) | 150.000 | 15% | 45:1 |
| URA | Energía uranio (ETF) | 100.000 | 10% | 5:1 |
| MP | Tierras raras (materiales críticos) | 150.000 | 15% | 10:1 |
| Cash | Reserva | 100.000 | 10% | — |

### Sleeve estable — $2.000.000 ARS (67%)
| Ticker | Rol | Peso ARS | % del sleeve | Ratio CEDEAR (BYMA) |
|---|---|---|---|---|
| O | REIT infraestructura/retail, paga mensual | 600.000 | 30% | 13:1 |
| KO | Dividend King (54 años) | 500.000 | 25% | 5:1 |
| PEP | Dividend King (54 años) | 400.000 | 20% | 18:1 |
| PG | Dividend King (57 años) | 300.000 | 15% | 15:1 |
| Cash | Reserva | 200.000 | 10% | — |

**Tickers a bajar en Claude Code:** `SMH NVDA CEG URA MP O KO PEP PG`

---

## 2. Qué se hizo antes (en este chat, con limitaciones)

- Se corrió un Monte Carlo basado en GARCH(1,1)/EWMA, pero el sandbox de este chat **no tiene acceso a Yahoo Finance ni Stooq** (whitelist de red restringida a pypi/github/etc.), así que yfinance falló.
- Se consiguieron series reales muy cortas (10 a 49 ruedas) vía scraping manual de TipRanks para NVDA, MP y O. Con esas muestras el ajuste GARCH(1,1) por máxima verosimilitud **degeneró** (α→0, β→1) porque 10-49 observaciones es insuficiente — se necesitan mínimo ~250 (idealmente 500-1000).
- Se resolvió con EWMA (RiskMetrics, λ=0,94) como aproximación estable, y correlaciones **asumidas a mano**, no estimadas de datos reales.

**Objetivo en Claude Code:** repetir esto bien, con series completas (2-3 años), GARCH(1,1) ajustado correctamente, y matriz de correlación estimada de los datos reales en vez de supuesta.

---

## 3. Prompt para pegar en Claude Code

```
Quiero un script de Python para un análisis de riesgo de portafolio con Monte Carlo + GARCH(1,1).

TICKERS Y PESOS (en ARS, capital total $3.000.000):
Sleeve riesgo ($1.000.000):
  SMH  -> 350000
  NVDA -> 150000
  CEG  -> 150000
  URA  -> 100000
  MP   -> 150000
  CASH -> 100000 (sin riesgo, no simular)

Sleeve estable ($2.000.000):
  O    -> 600000
  KO   -> 500000
  PEP  -> 400000
  PG   -> 300000
  CASH -> 200000 (sin riesgo, no simular)

PASOS QUE NECESITO (respetá este orden, es para un cliente que quiere ver las etapas separadas):

Etapa 1 - Descarga de datos:
- Usar yfinance para bajar precios diarios de cierre ajustado de los 8 tickers con retorno (SMH, NVDA, CEG, URA, MP, O, KO, PEP, PG), con al menos 3 años de historia (period="3y").
- Calcular retornos logarítmicos diarios en % para cada uno.
- Guardar todo en un DataFrame con fechas alineadas (dropear NaNs por feriados distintos si aplica).

Etapa 2 - Calibración GARCH(1,1) por activo:
- Usar la librería `arch` (arch_model) con mean='Zero', vol='Garch', p=1, q=1, dist='t' (Student-t, colas gordas) para cada uno de los 8 tickers.
- Reportar omega, alpha, beta, persistencia (alpha+beta), grados de libertad de la t, y la vol de largo plazo anualizada implícita (sqrt(omega/(1-alpha-beta)*252)).
- Si algún activo da persistencia >0.999 o parámetros degenerados, avisar explícitamente en el output (no ocultarlo) y usar como fallback EWMA con lambda=0.94.

Etapa 3 - Matriz de correlación:
- Calcular la matriz de correlación real de los retornos diarios de los 8 activos (no supuesta).
- Hacer la descomposición de Cholesky de esa matriz.

Etapa 4 - Simulación Monte Carlo:
- Horizonte: 126 ruedas (~6 meses).
- Al menos 10.000 escenarios.
- Para cada activo, simular la varianza día a día con la recursión GARCH ajustada en la Etapa 2 (o el fallback EWMA si degeneró), generando innovaciones t-Student correlacionadas vía la Cholesky de la Etapa 3.
- Drift = 0 (random walk puro, sin asumir retorno esperado) — explicar en un comentario por qué (no queremos apostar a una dirección, solo medir dispersión de riesgo).
- Aplicar los retornos simulados a los pesos en ARS de cada activo (los de CASH quedan fijos, sin simular).

Etapa 5 - Agregación y reporte:
- Sumar por sleeve (riesgo y estable) y total del portafolio, día a día.
- Calcular percentiles P5/P25/P50/P75/P95 del valor del portafolio en ARS a 6 meses, para el total y cada sleeve.
- Calcular probabilidad de terminar por debajo del capital inicial y VaR 95% (pérdida máxima esperada en el peor 5% de escenarios).
- Graficar un "fan chart": eje X = meses (0 a 6), líneas para P5/P25/P50/P75/P95 del valor total del portafolio.
- Imprimir una tabla resumen con todos los números.

Extra: si podés, comparar el resultado contra una corrida idéntica pero con retornos i.i.d. normales (sin GARCH, sin colas gordas) para mostrar cuánto cambia el VaR por usar colas gordas + volatilidad dinámica en vez de un modelo naive.

Aclaración: esto es para dimensionar riesgo, no para predecir dirección de mercado — no le agregues ningún tipo de "expected return" ni sesgo alcista/bajista al modelo salvo que yo lo pida explícitamente.
```

---

## 4. Contexto adicional para Claude Code (por si lo pide)

- **Moneda base del portafolio:** ARS. Los tickers cotizan en USD; para este análisis de riesgo relativo alcanza con simular en USD y aplicar directamente a los pesos en ARS (no hace falta modelar el tipo de cambio ARS/USD, salvo que se quiera agregar como fuente de riesgo adicional — el CCL ronda $1.575-1.580 al 13/7/2026, dato que puede quedar desactualizado).
- **Por qué drift=0:** decisión deliberada tomada en el análisis original, para que el resultado mida solo dispersión de riesgo y no una apuesta direccional disfrazada de forecast.
- **Por qué GARCH y no vol constante:** la volatilidad de activos como MP (tierras raras) o NVDA cambia mucho en el tiempo (clusters de volatilidad); un Monte Carlo con vol fija subestima las colas.
- Si Claude Code tiene acceso a internet real (a diferencia de este chat), yfinance debería funcionar sin problema y dar series limpias de 3+ años — ideal para que el GARCH converja bien, cosa que acá no se pudo lograr por la muestra corta.
