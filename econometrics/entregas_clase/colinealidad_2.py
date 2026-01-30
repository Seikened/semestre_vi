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

# y = Rendimiento (Variable Dependiente)
y = np.array([0.02, 0.01, -0.01, 0.03, 0.02])

# Variables Independientes
tasa_x1      = np.array([7.0, 7.2, 7.4, 7.1, 7.3])
cetes_x2     = np.array([6.9, 7.1, 7.3, 7.0, 7.2])
inflacion_x3 = np.array([4.1, 4.3, 4.5, 4.2, 4.4])

# Creamos la matriz X uniendo las 3 columnas
X = np.column_stack((tasa_x1, cetes_x2, inflacion_x3))

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
Construye la matriz de correlación entre las X.

Identifica pares con correlación alta.

¿Qué variables podrían causar multicolinealidad?

¿Cuál eliminarías y por qué?
"""


recta_pred = print_linear_expression(betas_list)
separador = "="*30

log.note(separador,"Estima el modelo de regresion multiple",separador)
log.info(print_betas(betas_list))
log.info(f"Mean Squared Error: {mse:.8f}")
log.info(f"Ecuación de la recta: {recta_pred}")



# Aquí llamamos a la nueva función de Colinealidad (VIF)

colinealidad(X, ["Tasa", "Cetes", "Inflacion"])

log.note("Respuesta:")
log.metric("Todas tienen una correlación muy alta")
log.metric("Podríamos eliminar Cetés o Inflación, ya que ambas están altamente correlacionadas con Tasa")
log.metric("Eliminaría Inflación, ya que Cetés es una tasa de interés y está más relacionada con la variable dependiente")