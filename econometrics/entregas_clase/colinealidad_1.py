from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import yfinance as yf  # <--- NUEVO IMPORT
from colorstreak import Logger as log
from sklearn import linear_model
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor # <--- NUEVO IMPORT

DATA_DIR = Path(__file__).resolve().parent / "data"
# dataset_path = DATA_DIR / "cardano.csv"

model = linear_model.LinearRegression()

# ====================== NUEVAS FUNCIONES DEFINIDAS ======================

def obtener_cierre_yahoo(ticker: str, start_date: str = "2023-01-01", end_date: str = None):
    """
    Descarga datos de Yahoo Finance y devuelve un DataFrame de Polars.
    """
    print(f"Descargando datos para: {ticker}...")
    
    # Descargar (Pandas)
    df_pd = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    df_pd.reset_index(inplace=True)
    
    # Limpiar nombres de columnas
    df_pd.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df_pd.columns]
    
    # Convertir a Polars
    df_pl = pl.from_pandas(df_pd)
    
    # Seleccionar fecha y cierre
    try:
        df_clean = df_pl.select([
            pl.col("Date").alias("fecha"),
            pl.col("Close").alias("cierre")
        ])
    except:
        df_clean = df_pl.select([
            pl.col("Date").alias("fecha"),
            pl.col("Adj Close").alias("cierre")
        ])

    return df_clean

def colinealidad(X: np.ndarray, feature_names: list):
    """
    Calcula el VIF y muestra la matriz de correlación en TEXTO (Terminal).
    """
    log.note("--- Análisis de Colinealidad (VIF) ---")
    
    # Agregamos constante temporalmente para que statsmodels calcule bien el VIF
    X_w_const = sm.add_constant(X)
    
    # VIF
    for i, nombre in enumerate(feature_names):
        # i+1 porque el índice 0 es la constante (intercepto)
        vif = variance_inflation_factor(X_w_const, i + 1)
        
        estado = "OK"
        if vif >= 5 and vif < 10:
            estado = "PRECAUCIÓN"
        elif vif >= 10:
            estado = "PELIGRO (Alta Colinealidad)"
            
        log.info(f"Variable {nombre:<10} | VIF: {vif:.4f} | {estado}")

    # Matriz de Correlación simple
    log.note("--- Matriz de Correlación ---")
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    print("          " + "".join([f"{n:>10}" for n in feature_names]))
    for i, row in enumerate(corr_matrix):
        print(f"{feature_names[i]:<10}" + "".join([f"{val:>10.2f}" for val in row]))


# ====================== TUS FUNCIONES EXISTENTES ======================

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
    X_con_constante = sm.add_constant(np.asarray(X))
    lm, p_lm, f_val, p_f = het_breuschpagan(residuals, X_con_constante)
    return float(p_lm)

def test_het_white(residuals: np.ndarray, X: np.ndarray) -> float:
    X_con_constante = sm.add_constant(np.asarray(X))
    lm, p_lm, f_val, p_f = het_white(residuals, X_con_constante)
    return float(p_lm)

def hipotesis_white(p_value: float):
    if p_value < 0.05:
        return "Rechazamos H0: Hay heterocedasticidad"
    else:
        return "No rechazamos H0: No hay heterocedasticidad"

def hipotesis_breusch_pagan(p_value: float):
    if p_value < 0.05:
        return "Rechazamos H0: Hay heterocedasticidad"
    else:
        return "No rechazamos H0: No hay heterocedasticidad"





# ====================== EJEMPLO DE REGRESION MULTIPLE ======================

# y = Gasto (Variable Dependiente)
y = np.array([9, 11, 13, 15, 17, 19])

# X = Ingreso (Variable Independiente)
ingreso = np.array([10, 12, 14, 16, 18, 20])

# Como en este ejercicio solo hay 1 variable (Ingreso), no usamos column_stack de 3.
# Hacemos reshape para que tenga formato de columna compatible con el modelo.
X = ingreso.reshape(-1, 1)

# ====================== SISTEMA DE REGRESION MULTIPLE ======================
model.fit(X, y)

beta_0 = model.intercept_

beta_1 = model.coef_[0]
betas_list = [beta_0, beta_1]
y_pred = model.predict(X)
residuals = y - y_pred
mse = np.mean(residuals ** 2)

#====================== Ejercicio de clase ======================
"""
Calcula la correlación entre Ingreso y Gasto.

¿Qué observas?

¿Crees que ambas variables aportan información distinta?

¿Hay riesgo de multicolinealidad?
"""


recta_pred = print_linear_expression(betas_list)
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas(betas_list))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")



# Aquí llamamos a la nueva función de Colinealidad (VIF)
x_n = X = np.column_stack((ingreso,y))
colinealidad(X, ["Ingreso", "Gasto"])

log.note("Respuesta:")
log.metric("1. La correlación entre Ingreso y Gasto es 1.0, lo que indica una relación lineal perfecta.")
log.metric("2. No, ambas variables no aportan información distinta debido a su alta correlación.")
log.metric("3. Sí, hay un riesgo significativo de multicolinealidad debido a la correlación perfecta entre las variables.")