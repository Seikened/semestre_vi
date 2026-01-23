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



# caso 4
tipo_cambio_x = np.array([18.5, 19.0, 19.8, 20.5, 21.0])
inflacion_y = np.array([3.2, 3.8, 4.5, 5.1, 5.6])

x = tipo_cambio_x
y = inflacion_y


X = x.reshape(-1, 1)

model.fit(X, y)

beta_0 = model.intercept_

beta_1 = model.coef_[0]
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== CASO 4 ======================
# Ejercicio:
"""
Construye el modelo.

¿Cómo interpretas una pendiente positiva?

¿Qué significa que el peso se deprecie?

¿Es correcto usar solo el tipo de cambio para explicar la inflación?
"""


recta_pred = print_linear_expression([beta_0, beta_1])
separador = "="*30

log.note(separador,"Construcción del modelo",separador)
log.info(print_betas([beta_0, beta_1]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"¿Cómo interpretas una pendiente positiva?",separador)
log.metric("Que los puntos o datos tiene relacion y no hay tanta dispersion")

log.note(separador,"¿Qué significa que el peso se deprecie?",separador)
log.metric("Que el valor del peso disminuye respecto a otras monedas")

log.note(separador,"¿Es correcto usar solo el tipo de cambio para explicar la inflación?",separador)
log.metric("No, ya que hay otros factores que influyen en la inflación, como la demanda, la oferta, las políticas monetarias, entre otros.")

