import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from colorstreak import Logger as log
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# ====================== UTILERÍAS (VISUALIZACIÓN) ======================

def graficar_heterocedasticidad(residuals, y_pred, x_var, x_name="Educación"):
    """
    Grafica: 
    1. Residuos vs Valores Ajustados (Detectar forma de 'abanico').
    2. Residuos vs Variable Independiente (Educación).
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(y_pred, residuals, alpha=0.6, color='blue', edgecolors='w')
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].set_title('Residuos vs Valores Ajustados')
    ax[0].set_xlabel('Salario Predicho')
    ax[0].set_ylabel('Residuos')

    ax[1].scatter(x_var, residuals, alpha=0.6, color='green', edgecolors='w')
    ax[1].axhline(0, color='red', linestyle='--')
    ax[1].set_title(f'Residuos vs {x_name}')
    ax[1].set_xlabel(x_name)
    
    plt.tight_layout()
    plt.show()

# ====================== UTILERÍAS (PRUEBAS ESTADÍSTICAS) ======================

def test_het_breusch_pagan(residuals, X_matriz):
    """Prueba Breusch-Pagan. H0: Homocedasticidad (Varianza constante)."""
    lm, p_lm, f_val, p_f = het_breuschpagan(residuals, X_matriz)
    msg = "Heterocedasticidad (Rechazo H0)" if p_lm < 0.05 else "Homocedasticidad (OK)"
    return p_lm, msg

def test_het_white(residuals, X_matriz):
    """Prueba de White (Más general). H0: Homocedasticidad."""
    lm, p_lm, f_val, p_f = het_white(residuals, X_matriz)
    msg = "Heterocedasticidad (Rechazo H0)" if p_lm < 0.05 else "Homocedasticidad (OK)"
    return p_lm, msg

def reporte_errores_robustos(model_ols):
    """Re-calcula los errores estándar usando la matriz robusta de White (HC3)"""
    robust_results = model_ols.get_robustcov_results(cov_type='HC3')
    return robust_results

# ====================== DATOS (SIMULACIÓN CORTE TRANSVERSAL) ======================

np.random.seed(99)
n_trabajadores = 100

educacion = np.random.uniform(5, 20, n_trabajadores)  # Años de educación
experiencia = np.random.uniform(0, 30, n_trabajadores) # Años de experiencia
error = np.random.normal(0, scale=educacion*150, size=n_trabajadores)

salario = 5000 + 1000 * educacion + 200 * experiencia + error

X_matrix = np.column_stack((educacion, experiencia))
X_sm = sm.add_constant(X_matrix) # Beta 0
y = salario

# ====================== EJECUCIÓN DEL CASO ======================

separador = "=" * 50

model = sm.OLS(y, X_sm).fit()
residuals = model.resid
y_pred = model.fittedvalues

log.note(separador, "1. Estimación MCO Inicial", separador)
log.info(f"Beta 0 (Constante): {model.params[0]:.2f}")
log.info(f"Beta 1 (Educación): {model.params[1]:.2f}")
log.info(f"Beta 2 (Experiencia): {model.params[2]:.2f}")

log.note("Generando gráficas de diagnóstico...")
graficar_heterocedasticidad(residuals, y_pred, educacion, "Educación")
log.note(separador, "2. Pruebas de Heterocedasticidad", separador)

bp_p, bp_msg = test_het_breusch_pagan(residuals, X_sm)
w_p, w_msg = test_het_white(residuals, X_sm)

log.metric("Breusch-Pagan p-value", f"{bp_p:.6f}")
log.info(f"Conclusión BP: {bp_msg}")
log.metric("White Test p-value", f"{w_p:.6f}")
log.info(f"Conclusión White: {w_msg}")


if bp_p < 0.05 or w_p < 0.05:
    log.error(separador, "¡Varianza no constante detectada! Corrigiendo...", separador)
    robust_model = reporte_errores_robustos(model)
    
    log.success("Modelo Reestimado con Errores Robustos (White/HC3):")
    log.info("\n" + str(robust_model.summary().tables[1]))
    
    log.note("Nota: Los coeficientes (Betas) NO cambian, pero los Errores Estándar y P-values son ahora confiables.")
else:
    log.success("La varianza es constante. El modelo MCO es válido.")