import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from colorstreak import Logger as log
from sklearn import linear_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf

# ====================== UTILERÍAS (VISUALIZACIÓN Y FORMATO) ======================

def print_betas(betas):
    terms = [f"Beta 0: {betas[0]:.2f}"]
    for i, beta in enumerate(betas[1:], start=1):
        terms.append(f"Beta {i}: {beta:.2f}")
    return ", ".join(terms)

def graficar_diagnostico_ts(residuals):
    """Grafica: 1. Residuos vs Tiempo (Estabilidad) y 2. ACF (Autocorrelación)"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Residuos vs Tiempo
    ax[0].plot(residuals, marker='o', linestyle='-', color='purple', alpha=0.6)
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].set_title('Residuos a través del Tiempo')
    # Autocorrelación (ACF)
    plot_acf(residuals, ax=ax[1], lags=15, title='Autocorrelación (ACF)')
    plt.tight_layout()
    plt.show()

# ====================== UTILERÍAS (PRUEBAS ESTADÍSTICAS) ======================

def test_autocorrelacion(residuals, lags=10):
    """Ejecuta Durbin-Watson y Ljung-Box. Retorna estadísticos y conclusión."""
    # 1. Durbin-Watson (Rango 0-4, Ideal ~2)
    dw = durbin_watson(residuals)
    dw_msg = "Positiva" if dw < 1.5 else "Negativa" if dw > 2.5 else "No hay"
    
    # 2. Ljung-Box (p < 0.05 indica autocorrelación)
    lb_df = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    lb_p = float(lb_df['lb_pvalue'].iloc[0])
    lb_msg = "Sí existe (Rechazo H0)" if lb_p < 0.05 else "No existe (Ruido Blanco)"
    
    return dw, dw_msg, lb_p, lb_msg

def modelo_gls_ar1(y, X):
    """Reestima usando Mínimos Cuadrados Generalizados (Corrige AR1)"""
    model_gls = sm.GLSAR(y, sm.add_constant(X), rho=1)
    return model_gls.iterative_fit(maxiter=5)

# ====================== DATOS (SIMULACIÓN 5 AÑOS / 60 MESES) ======================

np.random.seed(42)
n_meses = 60
ingreso_x = np.linspace(10, 50, n_meses)  # Ingreso creciente en el tiempo

u = np.zeros(n_meses)
for t in range(1, n_meses):
    u[t] = 0.8 * u[t-1] + np.random.normal(0, 2) 

consumo_y = 5 + 0.5 * ingreso_x + u

X = ingreso_x.reshape(-1, 1)
y = consumo_y

# ====================== EJECUCIÓN DEL CASO ======================

# 1. Estimación MCO
model = linear_model.LinearRegression()
model.fit(X, y)
betas = [model.intercept_, *model.coef_]
residuals = y - model.predict(X)
mse = np.mean(residuals ** 2)

separador = "=" * 50

log.note(separador, "1. Modelo MCO (Inicial)", separador)
log.info(print_betas(betas))
log.info(f"MSE: {mse:.4f}")

# 2. Diagnóstico Gráfico
log.note("Generando gráficas de Series de Tiempo...")
graficar_diagnostico_ts(residuals)

# 3. Pruebas Formales
dw_stat, dw_res, lb_p, lb_res = test_autocorrelacion(residuals)

log.note(separador, "2. Detección de Autocorrelación", separador)
log.metric("Durbin-Watson", f"{dw_stat:.4f} ({dw_res})")
log.metric("Ljung-Box p-value", f"{lb_p:.6f}")
log.info(f"Conclusión Ljung-Box: {lb_res}")

if lb_p < 0.05:
    log.error(separador, "¡Autocorrelación Detectada! Reestimando...", separador)
    res_gls = modelo_gls_ar1(y, X)
    
    log.success("Modelo (GLSAR - AR1):")
    beta0_new = res_gls.params[0]
    beta1_new = res_gls.params[1]
    log.info(f"Nuevos Betas: Beta 0: {beta0_new:.2f}, Beta 1: {beta1_new:.2f}")
    log.info(f"Rho (Autocorrelación estimada): {res_gls.model.rho[0]:.4f}")
else:
    log.success("El modelo es válido, no requiere corrección.")