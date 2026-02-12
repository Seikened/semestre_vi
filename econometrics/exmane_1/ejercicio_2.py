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



# Variable dependiente: Consumo (Y)
y = np.array([52, 54, 55, 56, 57, 58, 59, 60])

# Variables independientes (X)
x1 = np.array([80, 82, 84, 86, 88, 90, 92, 94])          # Ingreso
x2 = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4])  # Inflación
x3 = np.array([6.0, 6.2, 6.3, 6.5, 6.7, 6.9, 7.1, 7.3])  # Tasa

# Preparación de matrices para el modelo
X = np.column_stack((x1, x2, x3))

betas_list, y_pred, residuals, mse = ajuste_modelo(X, y)


"""
# Interpreta los coeficientes (ceteris paribus).
# Calcula el consumo estimado si: X1=86, X2=4.6, X3=6.5.
# Calcula VIF de cada X y concluye si hay multicolinealidad grave.
# Propón 1 acción correctiva.
"""



recta_pred = print_linear_expression(betas_list)
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas(betas_list))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")


log.note(separador,"Interpreta betas.",separador)
log.metric("beta 0", f"El consumo estimado cuando el ingreso es 0 es de {betas_list[0]:.2f} unidades.")
log.metric("beta 1", f"Por cada unidad adicional de ingreso, el consumo aumenta en promedio {betas_list[1]:.2f} unidades.")
log.metric("beta 2", f"Por cada unidad adicional de inflación, el consumo cambia en promedio {betas_list[2]:.2f} unidades.")
log.metric("beta 3", f"Por cada unidad adicional de tasa, el consumo cambia en promedio {betas_list[3]:.2f} unidades.")

log.note(separador,"Calcula el consumo estimado para X1=86, X2=4.6, X3=6.5.",separador)
X_nuevo = np.array([[86, 4.6, 6.5]])
consumo_estimado = betas_list[0] + betas_list[1]*X_nuevo[0, 0] + betas_list[2]*X_nuevo[0, 1] + betas_list[3]*X_nuevo[0, 2]

log.note(separador,"Calcula VIF de cada X y concluye si hay multicolinealidad grave.",separador)
colinealidad(X, ["Ingreso", "Inflación", "Tasa"])


log.note(separador,"Propón 1 acción correctiva.",separador)
log.metric("Una ccion a realizar es que se verifique que los datos estan correctamente recolectados y no hay errores de medicion que puedan estar generando colinealidad.")
