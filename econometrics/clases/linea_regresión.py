from colorstreak import Logger as log
import numpy as np
from sklearn import linear_model


model = linear_model.LinearRegression()



# Datos de la Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH)
ingreso_x = np.array([6, 8, 10, 12, 14]).reshape(-1, 1)
consumo_y = np.array([4.9, 5.6, 6.3, 7.0, 7.6])



model.fit(ingreso_x, consumo_y)


beta_0, beta_1 = model.intercept_, model.coef_[0]
y_pred = model.predict(ingreso_x)
y_real = consumo_y
mse = np.mean((y_pred - y_real) ** 2)
recta_pred = f"y = {beta_0:.2f} + {beta_1:.2f} * x"

log.info(f"Beta 0: {beta_0:.2f}, Beta 1: {beta_1:.2f}")
log.info(f"Mean Squared Error: {mse:.4f}")
log.info(f"Ecuación de la recta: {recta_pred}")


item = 2

x_obj = ingreso_x[item].reshape(-1, 1)  # X=10
y_pred_10 = model.predict(x_obj)[0]

y_obj = consumo_y[item]  # Y real para X=10 es 6.3
residuo = y_obj - y_pred_10

log.step(f"Para X={x_obj[0][0]}: Real={y_obj}, Predicho={y_pred_10:.2f}, Residuo={residuo:.2f}")