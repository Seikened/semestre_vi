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


def ajuste_modelo(X, y):
    """ Ajusta un modelo de regresión lineal y devuelve los coeficientes, predicciones, residuos y MSE. """
    model = linear_model.LinearRegression()
    model.fit(X, y)
    
    betas_list = [model.intercept_, *model.coef_]
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    return betas_list, y_pred, residuals, mse


# ======================== PRUEBAS ========================



# Ingreso (X)
x = np.array([6, 8, 10, 12, 14, 16])

# Consumo (Y)
y = np.array([5.1, 5.7, 6.4, 6.9, 7.7, 8.0])


betas_list, y_pred, residuals, mse = ajuste_modelo(x.reshape(-1, 1), y)



"""
Plantea el modelo.

Estima por MCO (β̂0 y β̂1).

Interpreta β̂0 y β̂1.

Calcula el consumo estimado para X=11.

Calcula el residuo para el hogar con X=10.
"""



recta_pred = print_linear_expression(betas_list)
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas(betas_list))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")


log.note(separador,"Interpreta β̂0 y β̂1.",separador)
log.metric("beta 0", f"El consumo estimado cuando el ingreso es 0 es de {betas_list[0]:.2f} unidades.")
log.metric("beta 1", f"Por cada unidad adicional de ingreso, el consumo aumenta en promedio {betas_list[1]:.2f} unidades.")

log.note(separador,"Calcula el consumo estimado para X=11.",separador)
ingreso_11 = 11
consumo_estimado_11 = betas_list[0] + betas_list[1] * ingreso_11
log.metric("Consumo Estimado para X=11", f"{consumo_estimado_11:.2f} unidades")

log.note(separador,"Calcula el residuo para el hogar con X=10.",separador)
ingreso_10 = 10
consumo_estimado_10 = betas_list[0] + betas_list[1] * ingreso_10
residuo_10 = y[2] - consumo_estimado_10  
log.metric("Residuo para X=10", f"{residuo_10:.2f} unidades")