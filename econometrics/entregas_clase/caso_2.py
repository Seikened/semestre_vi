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




# Datos de Horas de estudio (X)
horas_x = np.array([2, 4, 6, 8, 10])

# Datos de Calificación (Y)
calificacion_y = np.array([60, 68, 75, 82, 88])
y = calificacion_y


X = np.column_stack((horas_x, horas_x**2))

model.fit(X, y)

beta_0 = model.intercept_

beta_1, beta_2 = model.coef_
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== CASO 2 ======================
# Ejercicio:
"""
Construye el modelo.

¿Qué significa el intercepto 
β
0
​?

¿Es realista pensar que solo estudiar explica la calificación?

¿Qué otras variables podrían influir?
"""


recta_pred = print_linear_expression([beta_0, beta_1, beta_2])
separador = "="*30

log.note(separador,"Construcción del modelo",separador)
log.info(print_betas([beta_0, beta_1, beta_2]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note(separador,"¿Qué significa el intercepto β0?",separador)
log.metric("El intercepto β0 representa la calificación esperada cuando las horas de estudio son cero.")

log.note(separador,"¿Es realista pensar que solo estudiar explica la calificación?",separador)
log.metric("No, no es realista, necesitamos de más variables que expliquen la calificación.")

log.metric("Otras variables que podrían influir incluyen: calidad del sueño, nivel de estrés, métodos de estudio, asistencia a clases, entre otros.")
log.metric("- Sueño")
log.metric("- Estrés")
log.metric("- Métodos de estudio")
log.metric("- Asistencia a clases")
log.metric("- Calificación de exámenes anteriores")
