from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from colorstreak import Logger as log
from sklearn import linear_model
from statsmodels.stats.diagnostic import het_breuschpagan

DATA_DIR = Path(__file__).resolve().parent / "data"
dataset_path = DATA_DIR / "mex_fin(in).csv"



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


def test_breusch_pagan(residuals: np.ndarray, X: np.ndarray) -> float:
    """
    Retorna el p-value. Si es < 0.05, hay heterocedasticidad (problema).
    """
    # La prueba necesita intercepto explícito para calcularse bien
    X_con_constante = sm.add_constant(np.asarray(X))
    lm, p_lm, f_val, p_f = het_breuschpagan(residuals, X_con_constante)
    return float(p_lm)


def hipotesis_breusch_pagan(p_value: float):
    """
    H0: Homocedasticidad (varianza constante)
    H1: Heterocedasticidad (varianza no constante)
    """
    if p_value < 0.05:
        return "Rechazamos H0: Hay heterocedasticidad"
    else:
        return "No rechazamos H0: No hay heterocedasticidad"




# ======================
"""
┌────────────┬──────────┬───────┬──────┐
│ date       ┆ ipc      ┆ fix   ┆ rate │
│ ---        ┆ ---      ┆ ---   ┆ ---  │
│ str        ┆ f64      ┆ f64   ┆ f64  │
╞════════════╪══════════╪═══════╪══════╡
│ 2019-01-31 ┆ 44472.93 ┆ 19.54 ┆ 8.66 │
│ 2019-02-28 ┆ 45804.76 ┆ 19.98 ┆ 6.4  │
│ 2019-03-31 ┆ 46522.04 ┆ 19.5  ┆ 5.38 │
│ 2019-04-30 ┆ 45585.74 ┆ 19.7  ┆ 8.54 │
│ 2019-05-31 ┆ 45514.08 ┆ 19.26 ┆ 7.1  │
└────────────┴──────────┴───────┴──────┘
"""
df = pl.read_csv(dataset_path)

# Lo que hizo el profe (se hizo por que el data set n tiene la volatilidad del IPC)
df = df.with_columns([
    (pl.col("ipc") / pl.col("ipc").shift(1)).log().alias("ipc_ret"),
    (pl.col("fix") / pl.col("fix").shift(1)).log().alias("fix_ret")
])

# 2. Calculamos volatilidad 
df = df.with_columns(
    pl.col("ipc_ret").rolling_std(12).alias("vol_ipc")
)
# elimibar nulos
df = df.drop_nulls()

log.info(df.head())


# Datos individuales
ipc_y =df["ipc_ret"].to_numpy()
# Variables Independientes (X1, X2)
fix_x1 = df["fix_ret"].to_numpy()
rate_x2 = df["rate"].to_numpy()
vol_ipc_x3 = df["vol_ipc"].to_numpy()


X = np.column_stack((fix_x1, rate_x2, vol_ipc_x3))
y = ipc_y


# ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1, beta_2, beta_3 = model.coef_
betas_list = [beta_0, beta_1, beta_2, beta_3]
y_pred = model.predict(X)
residuals = y - y_pred
mse = np.mean(residuals ** 2)





#====================== Ejercicio de clase ======================
# Ejercicio:
"""
Ejercicio de clase
"""


recta_pred = print_linear_expression(betas_list)
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas(betas_list))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")

log.note("Prueba de Breusch-Pagan para heterocedasticidad")
p_value: float = test_breusch_pagan(residuals, X)
log.metric("p-value", f"{p_value:.6f}")
log.info(hipotesis_breusch_pagan(p_value))