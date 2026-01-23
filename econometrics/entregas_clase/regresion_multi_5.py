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
# Regresion lineal multiple - Ventas de una empresa y estrategia comercial

# Datos individuales
# Y = Ventas
ventas_y = np.array([120, 135, 140, 150, 145, 160])

# Variables Independientes (X1, X2, X3)
publicidad_x1 = np.array([10, 12, 14, 15, 13, 16])
precio_x2 = np.array([20, 19, 18, 18, 19, 17])
promociones_x3 = np.array([1, 1, 2, 2, 1, 3])


X = np.column_stack((publicidad_x1, precio_x2, promociones_x3))

y = ventas_y


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
Estima el modelo de regresión múltiple.

¿Qué variable representa una decisión estratégica?

¿Qué signo esperas para la publicidad?

¿Qué pasa si publicidad y promociones están correlacionadas?


"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2, beta_3])
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas([beta_0, beta_1, beta_2, beta_3]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"¿Qué variable representa una decisión estratégica?",separador)
log.metric("Publicidad y promociones son decisiones estratégicas")

log.note(separador,"¿Qué signo esperas para la publicidad?",separador)
log.metric("Se espera un signo positivo, ya que a mayor inversión en publicidad, se espera un aumento en las ventas.")

log.note(separador,"¿Qué pasa si publicidad y promociones están correlacionadas?",separador)
log.metric("Hablamos de que son colineales y no se puede estimar el modelo correctamente.")
