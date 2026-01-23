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
# Regresion lineal multiple - Salud pública: gasto y esperanza de vida

# Datos individuales
# Y = Esperanza vida
esperanza_y = np.array([72, 75, 78, 80, 82, 83])

# Variables Independientes (X1, X2, X3)
gasto_salud_x1 = np.array([5.0, 6.5, 7.0, 8.0, 9.0, 9.5])
pib_capita_x2 = np.array([9, 11, 14, 18, 22, 25])
medicos_x3 = np.array([1.8, 2.1, 2.5, 3.0, 3.5, 3.8])


X = np.column_stack((gasto_salud_x1, pib_capita_x2, medicos_x3))
y = esperanza_y


# ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1, beta_2, beta_3 = model.coef_
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== REGRESION MULTI VARIABLE 7 ======================
# Ejercicio:
"""
¿Qué variable controla el nivel de desarrollo?

Interpreta el coeficiente del gasto en salud.

¿Qué problema de endogeneidad puede existir?

¿Qué otra variable social incluirías?

"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2, beta_3])
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas([beta_0, beta_1, beta_2, beta_3]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"¿Qué variable controla el nivel de desarrollo?",separador)
log.metric("El PIB per cápita")

log.note(separador,"Interpreta el coeficiente del gasto en salud",separador)
log.metric("Entre mas se gaste en salud, mayor es la esperanza de vida")

log.note(separador,"¿Qué problema de endogeneidad puede existir?",separador)
log.metric("Podria existir una relación bidireccional, donde un mayor gasto en salud mejora la esperanza de vida, pero a su vez, una mayor esperanza de vida puede llevar a un mayor gasto en salud.")

log.note(separador,"¿Qué otra variable social incluirías?",separador)
log.metric("- Nivel educativo de la población")
log.metric("- Cantidad de ingenieros de un país")