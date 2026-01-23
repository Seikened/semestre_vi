from colorstreak import Logger as log
import numpy as np
from sklearn import linear_model


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

# ======================
# Regresion lineal multiple - Demanda de energía eléctrica (Economía pública)

# Datos individuales
# Y = Consumo
consumo_y = np.array([100, 102, 105, 98, 108, 110])

# Variables Independientes (X1, X2, X3)
pib_x1 = np.array([2.5, 2.0, 3.0, 1.5, 3.2, 3.5])
temperatura_x2 = np.array([22, 23, 24, 21, 25, 26])
precio_energia_x3 = np.array([1.8, 1.9, 1.7, 2.0, 1.6, 1.5])


X = np.column_stack((pib_x1, temperatura_x2, precio_energia_x3))
y = consumo_y


# ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1, beta_2, beta_3 = model.coef_
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== REGRESION MULTI VARIABLE 4 ======================
# Ejercicio:
"""
Interpreta el efecto de la temperatura.

¿Qué variable representa ingreso?

¿Qué signo esperas para el precio?

¿Qué problema podría surgir al usar datos anuales?


"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2, beta_3])
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas([beta_0, beta_1, beta_2, beta_3]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"Interpreta el efecto de la temperatura",separador)
log.metric("Por cada grado adicional en temperatura, el consumo de energía eléctrica cambia en promedio en", f"{beta_2:.2f} unidades, manteniendo constantes las demás variables.")

log.note(separador,"¿Qué variable representa ingreso?",separador)
log.metric("El PIB (X1) representa el ingreso.")

log.note(separador,"¿Qué signo esperas para el precio?",separador)
log.metric("No se podria ver la estacionalidad")