from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from colorstreak import Logger as log
from sklearn import linear_model
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

DATA_DIR = Path(__file__).resolve().parent / "data"
dataset_path = DATA_DIR / "mex_fin(in).csv"



model = linear_model.LinearRegression()



def print_linear_expression(betas):
    terms = [f"Beta 0: {betas[0]:.2f}"]
    for i, beta in enumerate(betas[1:], start=1):
        terms.append(f"{beta:.2f} * x{i}")
    return "y = " + " + ".join(terms)


def print_betas(betas):
    terms = [f"Beta 0: {betas[0]:.2f}"]
    for i, beta in enumerate(betas[1:], start=1):
        terms.append(f"Beta {i}: {beta:.2f}")
    return ", ".join(terms)


def test_breusch_pagan(residuals: np.ndarray, X: np.ndarray) -> float:
    """
    Retorna el p-value. Si es < 0.05, hay heterocedasticidad (problema).
    """
    # La prueba necesita intercepto explícito para calcularse bien
    X_con_constante = sm.add_constant(np.asarray(X))
    lm, p_lm, f_val, p_f = het_breuschpagan(residuals, X_con_constante)
    return float(p_lm)


def test_het_white(residuals: np.ndarray, X: np.ndarray) -> float:
    """
    Retorna el p-value. Si es < 0.05, hay heterocedasticidad (problema).
    """
    # La prueba necesita intercepto explícito para calcularse bien
    X_con_constante = sm.add_constant(np.asarray(X))
    lm, p_lm, f_val, p_f = het_white(residuals, X_con_constante)
    return float(p_lm)

def hipotesis_white(p_value: float):
    """
    H0: Homocedasticidad (varianza constante)
    H1: Heterocedasticidad (varianza no constante)
    """
    if p_value < 0.05:
        return "Rechazamos H0: Hay heterocedasticidad"
    else:
        return "No rechazamos H0: No hay heterocedasticidad"


def hipotesis_breusch_pagan(p_value: float):
    """
    H0: Homocedasticidad (varianza constante)
    H1: Heterocedasticidad (varianza no constante)
    """
    if p_value < 0.05:
        return "Rechazamos H0: Hay heterocedasticidad"
    else:
        return "No rechazamos H0: No hay heterocedasticidad"


def graficar_residuos(y_pred, residuals):
    """
    Genera un gráfico de dispersión de Residuos vs. Valores Ajustados.
    
    Parámetros:
    y_pred : array-like, valores predichos por el modelo.
    residuals : array-like, diferencia entre valores reales y predichos.
    """
    plt.figure(figsize=(10, 6))
    
    # Gráfico de dispersión
    plt.scatter(y_pred, residuals, color='blue', alpha=0.6, edgecolors='w', s=80)
    
    # Línea de referencia en 0 (donde deberían estar los residuos idealmente)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Referencia (0)')
    
    # Etiquetas y títulos
    plt.title('Gráfica de Residuos vs. Valores Ajustados (Homocedasticidad)', fontsize=14)
    plt.xlabel('Valores Ajustados (Predicciones)', fontsize=12)
    plt.ylabel('Residuos (Errores)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Mostrar gráfico
    plt.show()

# ======================

# Variable Dependiente (y): Rendimiento
rendimiento_y = np.array([0.01, 0.015, 0.018, 0.020, 0.025, -0.10, 0.15, -0.20, 0.30, -0.40])

# Variable Independiente (X): Riesgo
riesgo_x = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.5, 1.7, 2.0, 2.3, 2.8])

# (Opcional) Periodo, por si lo necesitas como referencia
periodo = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Preparación de matrices para el modelo
X = np.column_stack((riesgo_x,))
y = rendimiento_y

# ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1 = model.coef_[0]
betas_list = [beta_0, beta_1]
y_pred = model.predict(X)
residuals = y - y_pred
mse = np.mean(residuals ** 2)

#====================== Ejercicio de clase ======================
# Ejercicio:
"""
Estima el modelo.

Grafica los residuos.

¿Qué sucede con la varianza del error cuando el riesgo aumenta?

¿Este comportamiento es típico en finanzas?
"""


recta_pred = print_linear_expression(betas_list)
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas(betas_list))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note("Prueba de Breusch-Pagan para heterocedasticidad")
p_value: float = test_breusch_pagan(residuals, X)
log.metric("p-value", f"{p_value:.6f}")
log.info(hipotesis_breusch_pagan(p_value))

log.note(separador,"Qué sucede con la varianza del error cuando el riesgo aumenta?",separador)
log.metric("La varianza de los errores aumenta al aumentar el riesgo")

log.note(separador,"¿Este comportamiento es típico en finanzas?",separador)
log.metric("Sí, es típico observar heterocedasticidad en datos financieros")


graficar_residuos(y_pred, residuals)