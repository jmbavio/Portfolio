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

La Etapa 1 (descarga de precios) usa `requests` directo contra la API de
Yahoo Finance en vez de `yfinance`: `yfinance` depende de `curl_cffi` para
imitar un navegador, y ese cliente falla con "connection reset" detrás de
proxies que reterminan TLS (como los sandboxes de Claude Code). `requests`
con TLS estándar sí atraviesa ese tipo de proxy.
