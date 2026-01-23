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






# caso 3
publicidad_x = np.array([5, 7, 9, 11, 13])
ventas_y = np.array([20, 24, 28, 32, 36])

x = publicidad_x
y = ventas_y


X = publicidad_x.reshape(-1, 1)

model.fit(X, y)

beta_0 = model.intercept_

beta_1 = model.coef_[0]
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== CASO 3 ======================
# Ejercicio:
"""
Construye el modelo.

¿Qué representa 
β
1
​ para la empresa?

¿Puede existir un punto donde la publicidad ya no funcione igual?

¿Qué riesgos hay al usar solo una variable?
"""


recta_pred = print_linear_expression([beta_0, beta_1])
separador = "="*30

log.note(separador,"Construcción del modelo",separador)
log.info(print_betas([beta_0, beta_1]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"¿Qué representa β1 para la empresa?",separador)
log.metric("representa el cambio entre las ventas por la publicidad")

log.note(separador,"¿Puede existir un punto donde la publicidad ya no funcione igual?",separador)
log.metric("Si, puede saturar la publicidad y las ventas no aumenten igual")

log.note(separador,"¿Qué riesgos hay al usar solo una variable?",separador)
log.metric("No considerar otros factores que pueden influir en las ventas, como la calidad del producto, la competencia, la economía, entre otros.")

