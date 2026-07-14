"""
Analisis de riesgo de portafolio: Monte Carlo + GARCH(1,1) con colas gordas (t-Student).

Objetivo: dimensionar riesgo (dispersion de resultados posibles a 6 meses), NO predecir
direccion de mercado. Por eso el drift de la simulacion es 0 (random walk puro): así el
resultado mide solo volatilidad/correlacion, sin mezclar una apuesta direccional disfrazada
de forecast.

Requiere: yfinance, arch, pandas, numpy, scipy, matplotlib
    pip install yfinance arch pandas numpy scipy matplotlib
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from scipy.stats import norm, t as student_t

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuracion del portafolio
# ---------------------------------------------------------------------------

CAPITAL_TOTAL = 3_000_000  # ARS

SLEEVE_RIESGO = {
    "SMH": 350_000,
    "NVDA": 150_000,
    "CEG": 150_000,
    "URA": 100_000,
    "MP": 150_000,
}
CASH_RIESGO = 100_000

SLEEVE_ESTABLE = {
    "O": 600_000,
    "KO": 500_000,
    "PEP": 400_000,
    "PG": 300_000,
}
CASH_ESTABLE = 200_000

TICKERS = list(SLEEVE_RIESGO.keys()) + list(SLEEVE_ESTABLE.keys())
PESOS_ARS = {**SLEEVE_RIESGO, **SLEEVE_ESTABLE}

HORIZONTE_DIAS = 126  # ~6 meses habiles
N_SIMS = 10_000
EWMA_LAMBDA = 0.94
PERSISTENCIA_MAXIMA = 0.999
DIAS_POR_MES = HORIZONTE_DIAS // 6  # 21


# ---------------------------------------------------------------------------
# Etapa 1 - Descarga de datos
# ---------------------------------------------------------------------------

def descargar_retornos(tickers, period="3y"):
    """Baja precios de cierre ajustado y devuelve retornos log diarios en %."""
    precios = yf.download(tickers, period=period, auto_adjust=False, progress=False)["Adj Close"]
    precios = precios[tickers]  # orden estable
    retornos = np.log(precios / precios.shift(1)) * 100
    retornos = retornos.dropna(how="any")
    return retornos


# ---------------------------------------------------------------------------
# Etapa 2 - Calibracion GARCH(1,1) por activo (con fallback EWMA)
# ---------------------------------------------------------------------------

def calibrar_garch(retornos):
    """Ajusta GARCH(1,1)-t por activo. Si degenera, cae a EWMA(lambda=0.94)."""
    resultados = {}
    for ticker in retornos.columns:
        serie = retornos[ticker].values
        modelo = arch_model(serie, mean="Zero", vol="Garch", p=1, q=1, dist="t")
        fit = modelo.fit(disp="off")

        omega = fit.params["omega"]
        alpha = fit.params["alpha[1]"]
        beta = fit.params["beta[1]"]
        nu = fit.params["nu"]
        persistencia = alpha + beta

        degenerado = (
            persistencia > PERSISTENCIA_MAXIMA
            or alpha < 1e-6
            or omega < 1e-8
        )

        if degenerado:
            print(
                f"[AVISO] {ticker}: GARCH degenerado "
                f"(alpha={alpha:.4g}, beta={beta:.4g}, persistencia={persistencia:.4g}). "
                f"Usando fallback EWMA(lambda={EWMA_LAMBDA})."
            )
            var_incondicional = np.var(serie, ddof=1)
            resultados[ticker] = {
                "metodo": "EWMA",
                "var_incondicional": var_incondicional,
                "ultimo_sigma2": var_incondicional,
                "ultimo_residuo2": serie[-1] ** 2,
                "nu": None,
                "vol_largo_plazo_anual": np.sqrt(var_incondicional * 252),
            }
            continue

        vol_largo_plazo_anual = np.sqrt(omega / (1 - persistencia) * 252)
        cond_vol = fit.conditional_volatility
        resultados[ticker] = {
            "metodo": "GARCH",
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "nu": nu,
            "persistencia": persistencia,
            "vol_largo_plazo_anual": vol_largo_plazo_anual,
            "ultimo_sigma2": cond_vol[-1] ** 2,
            "ultimo_residuo2": serie[-1] ** 2,
        }

    print("\n=== Etapa 2: parametros GARCH(1,1) por activo ===")
    filas = []
    for ticker, r in resultados.items():
        if r["metodo"] == "GARCH":
            filas.append({
                "ticker": ticker,
                "metodo": r["metodo"],
                "omega": round(r["omega"], 4),
                "alpha": round(r["alpha"], 4),
                "beta": round(r["beta"], 4),
                "persistencia": round(r["persistencia"], 4),
                "nu (t-Student)": round(r["nu"], 2),
                "vol anual largo plazo %": round(r["vol_largo_plazo_anual"], 2),
            })
        else:
            filas.append({
                "ticker": ticker,
                "metodo": r["metodo"],
                "omega": None,
                "alpha": None,
                "beta": None,
                "persistencia": None,
                "nu (t-Student)": None,
                "vol anual largo plazo %": round(r["vol_largo_plazo_anual"], 2),
            })
    print(pd.DataFrame(filas).set_index("ticker").to_string())

    return resultados


# ---------------------------------------------------------------------------
# Etapa 3 - Matriz de correlacion real + Cholesky
# ---------------------------------------------------------------------------

def matriz_correlacion_cholesky(retornos):
    corr = retornos.corr()
    chol = np.linalg.cholesky(corr.values)
    print("\n=== Etapa 3: matriz de correlacion (retornos diarios reales) ===")
    print(corr.round(3).to_string())
    return corr, chol


# ---------------------------------------------------------------------------
# Etapa 4 - Simulacion Monte Carlo
# ---------------------------------------------------------------------------

def _innovaciones_correlacionadas_t(chol, n_sims, n_dias, nus):
    """
    Genera shocks estandarizados (media 0, varianza 1) correlacionados entre activos,
    con marginales t-Student de la nu propia de cada activo, via copula gaussiana:
    1) Z ~ Normal correlacionada (Cholesky de la matriz de correlacion real)
    2) U = Phi(Z)  (uniformes correlacionadas)
    3) shock_i = t_{nu_i}^-1(U_i), re-escalado a varianza 1
    """
    n_activos = chol.shape[0]
    z = np.random.standard_normal((n_sims, n_dias, n_activos))
    z_corr = z @ chol.T
    u = norm.cdf(z_corr)
    u = np.clip(u, 1e-6, 1 - 1e-6)

    shocks = np.empty_like(u)
    for i, nu in enumerate(nus):
        if nu is None or nu <= 2:
            shocks[:, :, i] = norm.ppf(u[:, :, i])
        else:
            escala = np.sqrt(nu / (nu - 2))  # normaliza t a varianza 1
            shocks[:, :, i] = student_t.ppf(u[:, :, i], df=nu) / escala
    return shocks


def simular_montecarlo(tickers, params_garch, chol, n_sims=N_SIMS, n_dias=HORIZONTE_DIAS,
                        usar_garch_dinamico=True):
    """
    Devuelve un array (n_sims, n_dias, n_activos) de retornos log simulados en %.
    Si usar_garch_dinamico=False, ignora GARCH/EWMA y usa vol constante + shocks normales
    i.i.d. (para la comparacion "naive" del punto Extra).
    """
    n_activos = len(tickers)
    nus = [params_garch[t].get("nu") for t in tickers]

    if not usar_garch_dinamico:
        shocks = np.random.standard_normal((n_sims, n_dias, n_activos))
        z_corr = shocks @ chol.T  # normales correlacionadas, ya con varianza 1
        vol_diaria = np.array([
            np.sqrt(params_garch[t]["vol_largo_plazo_anual"] ** 2 / 252) for t in tickers
        ])
        retornos = z_corr * vol_diaria  # drift = 0
        return retornos

    shocks = _innovaciones_correlacionadas_t(chol, n_sims, n_dias, nus)

    sigma2 = np.tile(
        np.array([params_garch[t]["ultimo_sigma2"] for t in tickers]), (n_sims, 1)
    )
    resid2 = np.tile(
        np.array([params_garch[t]["ultimo_residuo2"] for t in tickers]), (n_sims, 1)
    )

    retornos = np.empty((n_sims, n_dias, n_activos))
    for d in range(n_dias):
        for i, ticker in enumerate(tickers):
            p = params_garch[ticker]
            if p["metodo"] == "GARCH":
                sigma2[:, i] = p["omega"] + p["alpha"] * resid2[:, i] + p["beta"] * sigma2[:, i]
            else:  # EWMA
                sigma2[:, i] = EWMA_LAMBDA * sigma2[:, i] + (1 - EWMA_LAMBDA) * resid2[:, i]

        sigma = np.sqrt(sigma2)
        r_dia = shocks[:, d, :] * sigma  # drift = 0
        retornos[:, d, :] = r_dia
        resid2 = r_dia ** 2

    return retornos


# ---------------------------------------------------------------------------
# Etapa 5 - Agregacion y reporte
# ---------------------------------------------------------------------------

def agregar_portafolio(retornos_simulados, tickers):
    """
    A partir de retornos log simulados en % (n_sims, n_dias, n_activos), devuelve
    trayectorias de valor en ARS por sleeve y total (n_sims, n_dias+1), incluyendo t=0.
    """
    n_sims, n_dias, n_activos = retornos_simulados.shape
    factores = np.exp(retornos_simulados / 100)  # (1 + retorno) via log-retorno

    idx = {t: i for i, t in enumerate(tickers)}

    activo_valor = {}
    for t in tickers:
        peso_inicial = PESOS_ARS[t]
        camino = np.cumprod(factores[:, :, idx[t]], axis=1) * peso_inicial
        activo_valor[t] = np.concatenate(
            [np.full((n_sims, 1), peso_inicial), camino], axis=1
        )

    valores_riesgo = sum(activo_valor[t] for t in SLEEVE_RIESGO) + CASH_RIESGO
    valores_estable = sum(activo_valor[t] for t in SLEEVE_ESTABLE) + CASH_ESTABLE
    valores_totales = valores_riesgo + valores_estable

    return valores_riesgo, valores_estable, valores_totales


def reportar(valores_riesgo, valores_estable, valores_totales, etiqueta=""):
    capital_inicial = valores_totales[0, 0]
    percentiles = [5, 25, 50, 75, 95]

    def resumen(valores, nombre, capital_ini):
        finales = valores[:, -1]
        pct = {f"P{p}": np.percentile(finales, p) for p in percentiles}
        prob_perdida = np.mean(finales < capital_ini)
        var95 = capital_ini - np.percentile(finales, 5)
        return {
            "sleeve": nombre,
            "capital_inicial": capital_ini,
            **{k: round(v) for k, v in pct.items()},
            "prob_perdida_%": round(prob_perdida * 100, 2),
            "VaR_95_ARS": round(var95),
            "VaR_95_%": round(var95 / capital_ini * 100, 2),
        }

    filas = [
        resumen(valores_riesgo, "Riesgo", valores_riesgo[0, 0]),
        resumen(valores_estable, "Estable", valores_estable[0, 0]),
        resumen(valores_totales, "TOTAL", valores_totales[0, 0]),
    ]
    tabla = pd.DataFrame(filas).set_index("sleeve")
    print(f"\n=== Etapa 5: resumen a {HORIZONTE_DIAS} ruedas (~6 meses) {etiqueta} ===")
    print(tabla.to_string())
    return tabla


def graficar_fan_chart(valores_totales, ruta="fan_chart.png"):
    n_sims, n_dias_mas_1 = valores_totales.shape
    meses = np.arange(0, n_dias_mas_1, DIAS_POR_MES)
    if meses[-1] != n_dias_mas_1 - 1:
        meses = np.append(meses, n_dias_mas_1 - 1)
    x_meses = meses / DIAS_POR_MES

    percentiles = [5, 25, 50, 75, 95]
    curvas = {p: np.percentile(valores_totales[:, meses], p, axis=0) for p in percentiles}

    plt.figure(figsize=(9, 5.5))
    plt.fill_between(x_meses, curvas[5], curvas[95], alpha=0.15, color="steelblue", label="P5-P95")
    plt.fill_between(x_meses, curvas[25], curvas[75], alpha=0.30, color="steelblue", label="P25-P75")
    plt.plot(x_meses, curvas[50], color="navy", linewidth=2, label="Mediana (P50)")
    plt.axhline(valores_totales[0, 0], color="gray", linestyle="--", linewidth=1, label="Capital inicial")
    plt.xlabel("Meses")
    plt.ylabel("Valor del portafolio (ARS)")
    plt.title("Fan chart: proyeccion Monte Carlo del portafolio total a 6 meses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    print(f"\nFan chart guardado en: {ruta}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    print("=== Etapa 1: descarga de datos (yfinance, 3 años) ===")
    retornos = descargar_retornos(TICKERS, period="3y")
    print(f"Rango: {retornos.index.min().date()} a {retornos.index.max().date()} "
          f"({len(retornos)} ruedas)")

    params_garch = calibrar_garch(retornos)
    corr, chol = matriz_correlacion_cholesky(retornos)

    print("\n=== Etapa 4: simulacion Monte Carlo (GARCH/EWMA + t-Student correlacionada) ===")
    print(f"Horizonte: {HORIZONTE_DIAS} ruedas, {N_SIMS} escenarios, drift=0")
    ret_sim = simular_montecarlo(TICKERS, params_garch, chol, usar_garch_dinamico=True)
    v_riesgo, v_estable, v_total = agregar_portafolio(ret_sim, TICKERS)
    reportar(v_riesgo, v_estable, v_total, etiqueta="(GARCH/EWMA + t-Student)")
    graficar_fan_chart(v_total, ruta="fan_chart.png")

    print("\n=== Extra: comparacion contra modelo naive (normal i.i.d., vol constante) ===")
    ret_sim_naive = simular_montecarlo(TICKERS, params_garch, chol, usar_garch_dinamico=False)
    v_riesgo_n, v_estable_n, v_total_n = agregar_portafolio(ret_sim_naive, TICKERS)
    reportar(v_riesgo_n, v_estable_n, v_total_n, etiqueta="(naive: normal i.i.d.)")

    var95_dinamico = v_total[0, 0] - np.percentile(v_total[:, -1], 5)
    var95_naive = v_total_n[0, 0] - np.percentile(v_total_n[:, -1], 5)
    diff_pct = (var95_dinamico - var95_naive) / var95_naive * 100
    print(
        f"\nVaR 95% con GARCH+t: {var95_dinamico:,.0f} ARS | "
        f"VaR 95% naive (normal, vol fija): {var95_naive:,.0f} ARS | "
        f"diferencia: {diff_pct:+.1f}%"
    )


if __name__ == "__main__":
    main()
