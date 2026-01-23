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





# Datos de la columna Ingreso (X)
x = np.array([8, 10, 12, 14, 16])

# Datos de la columna Consumo (Y)
y = np.array([4.5, 5.2, 6.0, 6.8, 7.5])

X = np.column_stack((x, x**2))  # Incluye el término cuadrático


model.fit(X, y)

beta_0 = model.intercept_

beta_1, beta_2 = model.coef_
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== CASO 1 ======================
# Ejercicio:
"""
Construye el modelo.

¿Qué variable es Y y cuál es X?

¿Esperas que 
β
1
sea positiva o negativa?

Interpreta 
β
1
en palabras.

¿Qué factor importante no está incluido en el modelo?
"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2])
separador = "="*50

log.note(separador,"Construcción del modelo",separador)
log.info(print_betas([beta_0, beta_1, beta_2]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"Variables Y y X",separador)
log.metric("Variable Y: Consumo")
log.metric("Variable X: Ingreso")

log.note(separador,"Esperas que β1 sea positiva",separador)
log.metric(f"Beta 1: {beta_1:.2f}, debe ser positiva porque a mayor ingreso, mayor consumo.")

log.note(separador,"Interpretación de β1",separador)
log.metric(f"Beta 1: {beta_1:.2f}, indica que por cada unidad adicional de ingreso, el consumo aumenta en promedio {beta_1:.2f} unidades.")

log.note(separador,"Factor no incluido en el modelo",separador)
log.metric("Un factor importante que no está incluido en el modelo es el ahorro, ya que puede influir en el consumo de los individuos.")

