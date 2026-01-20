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

# Datos de la Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH)
ingreso_x1 = np.array([6, 8, 10, 12, 14]).reshape(-1, 1)
tasa_de_interes_x2 = np.array([3, 2.5, 2, 1.5, 1]).reshape(-1, 1)
X = np.hstack((ingreso_x1, tasa_de_interes_x2))
consumo_y = np.array([4.9, 5.6, 6.3, 7.0, 7.6])



model.fit(X, consumo_y)


beta_0 = model.intercept_
# Beta 1 es el consumo ante un aumento de ingreso, ceteris paribus
# Beta 2 es el consumo ante un aumento de la tasa de interés, manteniendo ingreso constante

beta_1, beta_2 = model.coef_
y_pred = model.predict(X)
y_real = consumo_y
mse = np.mean((y_pred - y_real) ** 2)


recta_pred = print_linear_expression([beta_0, beta_1, beta_2])
log.info(print_betas([beta_0, beta_1, beta_2]))
log.info(f"Mean Squared Error: {mse:.4f}")
log.info(f"Ecuación de la recta: {recta_pred}")


item = 2

x_obj = X[item].reshape(1, -1)  # X=10
y_pred_10 = model.predict(x_obj)[0]

y_obj = consumo_y[item]  # Y real para X=10 es 6.3
residuo = y_obj - y_pred_10

log.step(f"Para X={x_obj[0][0]}: Real={y_obj}, Predicho={y_pred_10:.2f}, Residuo={residuo:.2f}")