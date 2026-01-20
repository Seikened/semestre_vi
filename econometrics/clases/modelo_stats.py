import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1) Cargar datos y limpieza inicial
# ==========================================
file_path = "D:/Respaldo Jefferson/Educacion/2026/IBERO/Primavera 2026/Algoritmos Econometricos/mex_fin.csv"
df = pd.read_csv(file_path)

# Convertir a datetime y ordenar
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ==========================================
# 2) Feature Engineering (Rendimientos y Volatilidad)
# ==========================================
# En R: log(ipc / lag(ipc))
df['r_ipc'] = np.log(df['ipc'] / df['ipc'].shift(1))
df['r_tc']  = np.log(df['fix'] / df['fix'].shift(1))

# 3) Volatilidad móvil (Desviación estándar de 12 periodos)
# En R: rollapply(width=12, align="right")
df['vol_ipc'] = df['r_ipc'].rolling(window=12).std()

# Eliminar NAs generados por los lags y la ventana móvil
df2 = df.dropna(subset=['r_ipc', 'r_tc', 'rate', 'vol_ipc']).copy()

# ==========================================
# 4) Estimar modelo MCO (OLS)
# ==========================================
# Definir variables dependiente e independientes
X = df2[['r_tc', 'rate', 'vol_ipc']]
y = df2['r_ipc']

# ¡IMPORTANTE!: Statsmodels no agrega intercepto por defecto
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print("=== Resumen del Modelo MCO ===")
print(model.summary())

# ==========================================
# 5) Errores Estándar Robustos (HC1)
# ==========================================
# En R: vcovHC(m1, type = "HC1")
model_robust = model.get_robustcov_results(cov_type='HC1')
print("\n=== Coeficientes con Errores Robustos (HC1) ===")
print(model_robust.summary())

# ==========================================
# 6) Diagnósticos Básicos
# ==========================================

# A) Autocorrelación (Durbin-Watson)
dw_stat = durbin_watson(model.resid)
print(f"\nEstadístico Durbin-Watson: {dw_stat:.4f}")
# Nota: Cerca de 2 = No autocorrelación

# B) Heterocedasticidad (Breusch-Pagan)
# Retorna: LM stat, p-value LM, F stat, p-value F
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")

# C) Multicolinealidad (VIF)
# Creamos un DataFrame para mostrarlo bonito
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n=== Factor de Inflación de Varianza (VIF) ===")
print(vif_data)

# ==========================================
# 7) Gráficos de Diagnóstico (Réplica de plot(m1) en R)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# A) Residuals vs Fitted
axs[0, 0].scatter(model.fittedvalues, model.resid)
axs[0, 0].axhline(0, color='red', linestyle='--')
axs[0, 0].set_xlabel('Fitted values')
axs[0, 0].set_ylabel('Residuals')
axs[0, 0].set_title('Residuals vs Fitted')

# B) Q-Q Plot (Normalidad)
sm.qqplot(model.resid, line='45', ax=axs[0, 1])
axs[0, 1].set_title('Normal Q-Q')

# C) Scale-Location (Homocedasticidad visual)
# Raíz cuadrada de residuos estandarizados absolutos
resid_std = np.sqrt(np.abs(model.get_influence().resid_studentized_internal))
axs[1, 0].scatter(model.fittedvalues, resid_std)
axs[1, 0].set_xlabel('Fitted values')
axs[1, 0].set_ylabel('sqrt(|Standardized residuals|)')
axs[1, 0].set_title('Scale-Location')

# D) Residuals vs Leverage
# (Versión simplificada del plot de Cook)
influence = model.get_influence()
leverage = influence.hat_matrix_diag
resid_studentized = influence.resid_studentized_internal
axs[1, 1].scatter(leverage, resid_studentized)
axs[1, 1].set_xlabel('Leverage')
axs[1, 1].set_ylabel('Standardized Residuals')
axs[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.show()

# ==========================================
# 8) Tabla "Tidy" de coeficientes ordenada
# ==========================================
tidy_df = pd.DataFrame({
    'term': X.columns,
    'estimate': model.params.values,
    'std.error': model.bse.values,
    'statistic': model.tvalues.values,
    'p.value': model.pvalues.values
}).sort_values('p.value')

print("\n=== Tabla Tidy Ordenada por P-Value ===")
print(tidy_df)