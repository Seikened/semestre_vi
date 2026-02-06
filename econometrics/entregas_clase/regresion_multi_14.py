import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from colorstreak import Logger as log
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ====================== UTILERÍAS (VISUALIZACIÓN Y DIAGNÓSTICO) ======================

def graficar_correlacion(df_X):
    """Genera un mapa de calor simple de la matriz de correlación."""
    corr = df_X.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)
    
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", 
                    ha='center', va='center', color='black')
            
    ax.set_title("Matriz de Correlación")
    plt.show()

def calcular_vif(X_df):
    """
    Calcula el Factor de Inflación de Varianza (VIF).
    VIF > 10 indica Multicolinealidad Severa.
    """
    X_vals = X_df.values.astype(float)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vals, i) 
                       for i in range(X_vals.shape[1])]
    
    return vif_data

# ====================== DATOS (SIMULACIÓN MULTICOLINEALIDAD) ======================

np.random.seed(101)
n_empresas = 100

# 1. Variable "Activos" (Variable base)
activos = np.random.normal(1000, 200, n_empresas)

# 2. Generamos MULTICOLINEALIDAD:
ventas = activos * 1.5 + np.random.normal(0, 50, n_empresas)  # Corr muy alta con activos
capital_trabajo = activos * 0.3 + np.random.normal(0, 20, n_empresas) # Corr alta

roa = 5 + 0.001*activos + 0.002*ventas - 0.005*capital_trabajo + np.random.normal(0, 1, n_empresas)

df_X = pd.DataFrame({
    'Constante': 1.0, 
    'Activos': activos,
    'Ventas': ventas,
    'CapTrabajo': capital_trabajo
})
y = roa

# ====================== EJECUCIÓN DEL CASO ======================

separador = "=" * 50

model = sm.OLS(y, df_X).fit()

log.note(separador, "1. Modelo MCO (Con Multicolinealidad)", separador)
log.info(f"R-squared: {model.rsquared:.4f}")
print(model.summary().tables[1]) 

log.note("\nGenerando Matriz de Correlación...")
graficar_correlacion(df_X.drop(columns=['Constante'])) 

log.note(separador, "2. Factor de Inflación de Varianza (VIF)", separador)
vif_df = calcular_vif(df_X)


for index, row in vif_df.iterrows():
    val = row['VIF']
    var = row['Variable']
    if var != 'Constante':
        if val > 10:
            log.error(f"{var}: {val:.2f} -> ¡Multicolinealidad Severa!")
        else:
            log.success(f"{var}: {val:.2f} -> Aceptable")

log.note(separador, "3. Solución: Transformación a Ratios", separador)

rotacion_activos = ventas / activos
df_X_nuevo = pd.DataFrame({
    'Constante': 1.0,
    'Rotacion_Activos': rotacion_activos,
    'CapTrabajo_Activos': capital_trabajo / rotacion_activos
})

model_fix = sm.OLS(y, df_X_nuevo).fit()

log.info("Modelo Reestimado (Usando Ratios):")
print(model_fix.summary().tables[1])
log.success("Al usar ratios, eliminamos la colinealidad por 'tamaño de empresa'.")