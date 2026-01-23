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
# Regresion lineal multiple - Inflación y factores macroeconómicos

# Datos individuales
# Y = Inflación
inflacion_y = np.array([3.2, 3.8, 4.5, 5.1, 4.0, 3.6])

# Variables Independientes (X1, X2, X3)
tipo_cambio_x1 = np.array([18.5, 19.0, 19.8, 20.5, 20.0, 19.5])
tasa_interes_x2 = np.array([6.0, 6.5, 7.0, 7.5, 7.0, 6.5])
crec_pib_x3 = np.array([2.5, 2.0, 3.0, 1.5, 2.8, 3.2])


X = np.column_stack((tipo_cambio_x1, tasa_interes_x2, crec_pib_x3))

y = inflacion_y



# ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1, beta_2, beta_3 = model.coef_
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== REGRESION MULTI VARIABLE 3 ======================
# Ejercicio:
"""
Estima el modelo de regresión múltiple.

Interpreta el efecto del tipo de cambio.

¿Qué variable controla la demanda agregada?

¿Qué problemas econométricos pueden surgir?


"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2, beta_3])
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas([beta_0, beta_1, beta_2, beta_3]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"Interpreta el efecto del tipo de cambio",separador)
log.metric("A medida que la tasa de cambio aumenta, la inflación tiende a aumentar, lo que indica una relación positiva entre ambas variables.")

log.note(separador,"¿Qué variable controla la demanda agregada?",separador)
log.metric("La tasa de interés controla la demanda agregada, ya que afecta el costo del crédito y el consumo.")

log.note(separador,"¿Qué problemas econométricos pueden surgir?",separador)
log.metric("Que los datos no sean estacionarios, lo que puede llevar a resultados engañosos en la regresión.")

