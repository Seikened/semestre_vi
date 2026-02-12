from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import yfinance as yf 
from colorstreak import Logger as log
from sklearn import linear_model
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ======================
# Funcion para extraer datos
# ======================
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


# ======================
# Funciones para econometría
# ======================
def print_linear_expression(betas):
    """ Se asume que betas[0] es el intercepto, y el resto son los coeficientes de las variables X1, X2, ... """
    terms = [f"Beta 0: {betas[0]:.2f}"]
    for i, beta in enumerate(betas[1:], start=1):
        terms.append(f"{beta:.2f} * x{i}")
    return "y = " + " + ".join(terms)


def print_betas(betas):
    """ Se asume que betas[0] es el intercepto, y el resto son los coeficientes de las variables X1, X2, ... """
    terms = [f"Beta 0: {betas[0]:.2f}"]
    for i, beta in enumerate(betas[1:], start=1):
        terms.append(f"Beta {i}: {beta:.2f}")
    return ", ".join(terms)

def test_breusch_pagan(residuals: np.ndarray, X: np.ndarray) -> float:
    """ Se usa la función de statsmodels para realizar la prueba de Breusch-Pagan. Retorna el p-value. Si es < 0.05, hay heterocedasticidad (problema). """
    X_con_constante = sm.add_constant(np.asarray(X))
    lm, p_lm, f_val, p_f = het_breuschpagan(residuals, X_con_constante)
    return float(p_lm)

def test_het_white(residuals: np.ndarray, X: np.ndarray) -> float:
    """ Se usa la función de statsmodels para realizar la prueba de White. Retorna el p-value. Si es < 0.05, hay heterocedasticidad (problema). """
    X_con_constante = sm.add_constant(np.asarray(X))
    lm, p_lm, f_val, p_f = het_white(residuals, X_con_constante)
    return float(p_lm)

def hipotesis_white(p_value: float):
    """ H0: Homocedasticidad (varianza constante) H1: Heterocedasticidad (varianza no constante) """
    if p_value < 0.05:
        return "Rechazamos H0: Hay heterocedasticidad"
    else:
        return "No rechazamos H0: No hay heterocedasticidad"

def hipotesis_breusch_pagan(p_value: float):
    """ H0: Homocedasticidad (varianza constante) H1: Heterocedasticidad (varianza no constante) """
    if p_value < 0.05:
        return "Rechazamos H0: Hay heterocedasticidad"
    else:
        return "No rechazamos H0: No hay heterocedasticidad"

def colinealidad(X: np.ndarray, feature_names: list):
    """
    Calcula el VIF y muestra la matriz de correlación en TEXTO (Terminal).
    """
    log.note("--- Análisis de Colinealidad (VIF) ---")
    
    X_w_const = sm.add_constant(X)
    
    for i, nombre in enumerate(feature_names):
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


# ======================

# Variable Dependiente (y): Consumo
consumo_y = np.array([3.2, 3.8, 4.1, 4.5, 5.0, 7.5, 8.9, 11.2, 13.8, 18.5])

# Variable Independiente (X): Ingreso
ingreso_x = np.array([5, 6, 7, 8, 9, 15, 18, 22, 25, 30])

# (Opcional) Identificador de Hogar
hogar = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Preparación de matrices para el modelo
X = np.column_stack((ingreso_x,))
y = consumo_y

# ======================


def ajuste_modelo(X, y):
    """ Ajusta un modelo de regresión lineal y devuelve los coeficientes, predicciones, residuos y MSE. """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    
    betas_list = [model.intercept_, *model.coef_]
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    return betas_list, y_pred, residuals, mse