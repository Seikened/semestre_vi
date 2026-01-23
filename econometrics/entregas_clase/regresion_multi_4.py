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
# Regresion lineal multiple - Desempeño académico universitario

# Datos individuales
# Y = Promedio
promedio_y = np.array([78, 82, 75, 88, 90, 85])

# Variables Independientes (X1, X2, X3)
horas_estudio_x1 = np.array([10, 12, 8, 15, 16, 14])
asistencia_x2 = np.array([80, 85, 78, 90, 92, 88])
horas_trabajo_x3 = np.array([20, 15, 25, 10, 8, 12])


X = np.column_stack((horas_estudio_x1, asistencia_x2, horas_trabajo_x3))

y = promedio_y



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

¿Qué variable esperas que tenga efecto negativo?

¿Es razonable una relación lineal?

¿Qué variable faltante podría mejorar el modelo?


"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2, beta_3])
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas([beta_0, beta_1, beta_2, beta_3]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"¿Qué variable esperas que tenga efecto negativo?",separador)
log.metric("Beta 3 (horas de trabajo): Se espera que tenga un efecto negativo pero no lo tiene....")

log.note(separador,"¿Es razonable una relación lineal?",separador)
log.metric("Sí, es razonable ya que a mayor horas de estudio y asistencia, el promedio debería aumentar de manera lineal.")


log.note(separador,"¿Qué variable faltante podría mejorar el modelo?",separador)
log.metric("- Horas de sueño")
log.metric("- Nivel socioeconómico")
log.metric("- Estrés académico")
log.metric("- Métodos de estudio")
