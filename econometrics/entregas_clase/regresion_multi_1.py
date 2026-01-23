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


# Regresion multi variable 1

# Datos individuales
# Y = Consumo
consumo_y = np.array([50, 52, 55, 53, 56, 58])

# Variables Independientes (X1, X2, X3)
ingreso_x1 = np.array([80, 82, 85, 83, 88, 90])
inflacion_x2 = np.array([4.0, 4.5, 4.0, 5.0, 4.2, 4.0])
tasa_x3 = np.array([6.0, 6.5, 6.0, 7.0, 6.2, 6.0])


X = np.column_stack((ingreso_x1, inflacion_x2, tasa_x3))

y = consumo_y



# ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1, beta_2, beta_3 = model.coef_
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== REGRESION MULTI VARIABLE 1 ======================
# Ejercicio:
"""
Estima el modelo de regresión múltiple.

¿Qué variable explica mejor el consumo?

Interpreta 
β
2
β2​ en términos económicos.

¿Qué problema habría si se omite la inflación?
"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2, beta_3])
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas([beta_0, beta_1, beta_2, beta_3]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"Qué variable explica mejor el consumo?",separador)
log.metric(f"Beta 1 (ingreso) es la variable que más impacta en el consumo que es: {beta_1:.2f} por cada unidad adicional en ingreso.")

log.note(separador,"Interpreta β2 en términos económicos",separador)
log.metric("A mayor inflación menor poder adquisitivo, por lo que el consumo disminuye.")

log.note(separador,"¿Qué problema habría si se omite la inflación?",separador)
log.metric("Se incurre en un sesgo de variable omitida, ya que la inflación afecta tanto al consumo como al ingreso.")


