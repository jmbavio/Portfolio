# Portfolio
Un poco de finanzas

## Análisis de riesgo Monte Carlo + GARCH

`montecarlo_garch.py` implementa el análisis descrito en
[`portafolio_IA_montecarlo_claude_code.md`](portafolio_IA_montecarlo_claude_code.md):
Monte Carlo con GARCH(1,1)-t (fallback EWMA si el ajuste degenera) sobre el
portafolio de 9 tickers, drift=0, horizonte de 6 meses.

```bash
pip install -r requirements.txt
python montecarlo_garch.py
```

Requiere acceso real a Yahoo Finance (vía `yfinance`) para la Etapa 1; en
sandboxes con salida de red restringida la descarga fallará, tal como se
documenta en el `.md`.

