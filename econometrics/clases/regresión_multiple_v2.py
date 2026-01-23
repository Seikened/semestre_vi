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



# Variable dependiente: Rend_IPC (Y)
Y = np.array([0.012, 0.008, -0.004, 0.015, 0.010, 0.006])

# Variable independiente 1: Rend_TC (X1)
X1 = np.array([0.010, 0.015, 0.020, -0.005, 0.000, 0.008])

# Variable independiente 2: Tasa (%) (X2)
X2 = np.array([7.25, 7.50, 7.75, 7.00, 7.25, 7.40])

# Variable independiente 3: Volatilidad (X3)
X3 = np.array([0.020, 0.030, 0.045, 0.018, 0.022, 0.028])

X = np.column_stack((X1, X2, X3))



model.fit(X, Y)

beta_0 = model.intercept_

beta_1, beta_2, beta_3 = model.coef_
y_pred = model.predict(X)
y_real = Y
mse = np.mean((y_pred - y_real) ** 2)


recta_pred = print_linear_expression([beta_0, beta_1, beta_2, beta_3])
log.note("Estima el modelo por MCO")
log.info(print_betas([beta_0, beta_1, beta_2, beta_3]))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note("¿Que interpreta el signo de cada coeficiente?")
log.metric("Beta 1: A mayor rendimiento del tipo de cambio, mayor rendimiento del IPC.")
log.metric("Beta 2: A mayor tasa, menor rendimiento del IPC.")
log.metric("Beta 3: A mayor volatilidad, menor rendimiento del IPC")


log.note("La variable que parece tener mayor impacto es la volatilidad ")

log.note("¿Esperas problemas de autocorrelación? ¿Por qué?")
log.metric("Sí, porque los rendimientos financieros suelen mostrar autocorrelación debido a factores como tendencias del mercado y eventos económicos que afectan múltiples periodos.")



