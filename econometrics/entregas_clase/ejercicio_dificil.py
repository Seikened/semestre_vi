from colorstreak import Logger as log
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt   


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
# Regresion lineal simple - Hotel y Restaurante

# X = Habitaciones ocupadas (Variable Independiente)
habitaciones_x = np.array([
    23, 47, 21, 39, 37, 29, 23, 44, 45, 16, 30, 42, 54, 
    27, 34, 15, 19, 38, 44, 47, 43, 38, 51, 61, 39
])

# Y = Ingreso (Variable Dependiente)
ingreso_y = np.array([
    1452, 1361, 1426, 1470, 1456, 1430, 1354, 1442, 1394, 1459, 1399, 1458, 1537,
    1425, 1445, 1439, 1348, 1450, 1431, 1446, 1485, 1405, 1461, 1490, 1426
])


# Como solo es UNA variable X, usamos reshape en lugar de column_stack
X = habitaciones_x.reshape(-1, 1)
y = ingreso_y

# ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1  = model.coef_[0]
y_pred = model.predict(X)
y_real = y
mse = np.mean((y_pred - y_real) ** 2)





#====================== REGRESION SIMPLE DIFICIL ======================
# Ejercicio:
"""
a) ¿Parece que aumenta el ingreso por desayunos a medida que aumenta el número de habitaciones ocupadas? Trace un diagrama de dispersión para apoyar su conclusión.
b) Determine el coeficiente de correlación entre las dos variables. Interprete el valor.
c) ¿Es razonable concluir que hay una relación positiva entre ingreso y habitaciones ocupadas?
Utilice el nivel de significancia 0.10.
d) ¿Qué porcentaje de la variación de los ingresos del restaurante se contabilizan por el número
de habitaciones ocupadas?
"""


recta_pred = print_linear_expression([beta_0, beta_1])
separador = "="*30

log.note(separador,"Estima el modelo de regresion simple",separador)
log.info(print_betas([beta_0, beta_1]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note("a) ¿Parece que aumenta el ingreso por desayunos a medida que aumenta el número de habitaciones ocupadas? Trace un diagrama de dispersión para apoyar su conclusión.",separador)
log.metric("No parece que aumenta el ingreso, hay mucha dispersión en los datos.")

plt.scatter(habitaciones_x, ingreso_y, color='blue', label='Datos reales')
plt.plot(habitaciones_x, y_pred, color='red', label='Recta de predicción')
plt.xlabel('Habitaciones Ocupadas')
plt.ylabel('Ingreso por Desayunos')
plt.title('Diagrama de Dispersión: Ingreso vs Habitaciones Ocupadas')
plt.legend()
plt.grid()
plt.show()


log.note("b) Determine el coeficiente de correlación entre las dos variables. Interprete el valor.",separador)
correlation_matrix = np.corrcoef(habitaciones_x, ingreso_y)
correlation_xy = correlation_matrix[0,1]
log.metric("El coeficiente de correlación es", f"{correlation_xy:.4f}. Esto indica una correlación positiva débil entre las dos variables.")

log.note("c) ¿Es razonable concluir que hay una relación positiva entre ingreso y habitaciones ocupadas? Utilice el nivel de significancia 0.10.",separador)
log.metric("No es razonable concluir que hay una relación positiva, ya que el coeficiente de correlación es bajo y los datos están muy dispersos.")

log.note("d) ¿Qué porcentaje de la variación de los ingresos del restaurante se contabilizan por el número de habitaciones ocupadas?",separador)
r_squared = correlation_xy ** 2
log.metric("El porcentaje de variación de los ingresos explicado por el número de habitaciones ocupadas es", f"{r_squared*100:.2f}%.")


