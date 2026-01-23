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



# caso 5
tamano_x = np.array([60, 80, 100, 120, 140])
precio_y = np.array([900, 1150, 1400, 1650, 1900])

x = tamano_x
y = precio_y


X = x.reshape(-1, 1)

model.fit(X, y)

beta_0 = model.intercept_

beta_1 = model.coef_[0]
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== CASO 5 ======================
# Ejercicio:
"""
Contruye el modelo

¿Qué indica 
β
1
​?

¿Es razonable que dos casas del mismo tamaño tengan distinto precio?

¿Qué variables faltan?
"""


recta_pred = print_linear_expression([beta_0, beta_1])
separador = "="*30

log.note(separador,"Construcción del modelo",separador)
log.info(print_betas([beta_0, beta_1]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"¿Qué indica β1?",separador)
log.metric("Beta 1: Que a mayor tamaño de la casa, mayor precio.")

log.note(separador,"¿Es razonable que dos casas del mismo tamaño tengan distinto precio?",separador)
log.metric("Sí, porque pueden influir otros factores como ubicación, estado de la casa, número de habitaciones, indice de seguridad,etc.")

log.note(separador,"¿Qué variables faltan?",separador)
log.metric("- Ubicación")
log.metric("- Indice de seguridad")
log.metric("- Condición de la casa")
log.metric("- Número de habitaciones")
log.metric("- Edad de la casa")
log.metric("- Amenidades cercanas")