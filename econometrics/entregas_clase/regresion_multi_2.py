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

# Regresion lineal multiple - Remesas y crecimiento económico

# Datos individuales
# Y = Crec_PIB
crec_pib_y = np.array([2.1, 1.8, 2.5, 1.2, 2.8, 3.0])

# Variables Independientes (X1, X2, X3)
remesas_x1 = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
inversion_x2 = np.array([20, 19, 21, 18, 22, 23])
apertura_x3 = np.array([60, 58, 62, 57, 63, 65])


X = np.column_stack((remesas_x1, inversion_x2, apertura_x3))

y = crec_pib_y



# ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1, beta_2, beta_3 = model.coef_
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== REGRESION MULTI VARIABLE 2 ======================
# Ejercicio:
"""
Estima el modelo de regresión múltiple.

Interpreta el coeficiente de las remesas ceteris paribus.

¿Cuál variable parece más relevante para el crecimiento?

¿Qué problema econométrico podría existir entre remesas y crecimiento?

"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2, beta_3])
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas([beta_0, beta_1, beta_2, beta_3]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"Interpreta el coeficiente de las remesas ceteris paribus",separador)
log.metric(f"Beta 1 (remesas) : es una variable con {beta_1:.2f} que casi no impacta por el dato que esta en -0.10")

log.note(separador,"¿Cuál variable parece más relevante para el crecimiento?",separador)
log.metric("Beta 2: (inversión) es la variable que más impacta en el crecimiento económico.")

log.note(separador,"¿Qué problema econométrico podría existir entre remesas y crecimiento?",separador)
log.metric("Podría existir un problema de que no significa causalidad entre remesas y crecimiento económico. Aparte que tienen el mismo factor (beta 2 y beta 3)")


