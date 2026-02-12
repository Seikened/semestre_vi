from extraccion_datos import get_columnas_post, get_columnas_pre
from funciones import AnalisisEconometrico
import numpy as np
import statsmodels.api as sm

# ============= Análisis Pre-Choque (2015-2019) =============
print("Analisis Pre-Choque (2015-2019)")
inflacion_pre, tiie_pre, igae_pre, fix_pre, petroleo_pre = get_columnas_pre()
X_pre = np.column_stack((tiie_pre, igae_pre, fix_pre, petroleo_pre))
y_pre = inflacion_pre
AnalisisEconometrico.estimar_mco(X_pre, y_pre, feature_names=['TIIE', 'IGAE', 'FIX', 'Petróleo'], verbose=True)
AnalisisEconometrico.test_heterocedasticidad(AnalisisEconometrico.estimar_mco(X_pre, y_pre)[0].resid, sm.add_constant(X_pre))
AnalisisEconometrico.test_autocorrelacion(AnalisisEconometrico.estimar_mco(X_pre, y_pre)[0].resid)
AnalisisEconometrico.test_multicolinealidad(X_pre, ['TIIE', 'IGAE', 'FIX', 'Petróleo'])


# ============= Análisis Post-Choque (2020-2024) =============
print("\nAnalisis Post-Choque (2020-2024)")
inflacion_post, tiie_post, igae_post, fix_post, petroleo_post = get_columnas_post()
X_post = np.column_stack((tiie_post, igae_post, fix_post, petroleo_post))
y_post = inflacion_post
AnalisisEconometrico.estimar_mco(X_post, y_post, feature_names=['TIIE', 'IGAE', 'FIX', 'Petróleo'], verbose=True)
AnalisisEconometrico.test_heterocedasticidad(AnalisisEconometrico.estimar_mco(X_post, y_post)[0].resid, sm.add_constant(X_post))
AnalisisEconometrico.test_autocorrelacion(AnalisisEconometrico.estimar_mco(X_post, y_post)[0].resid)
AnalisisEconometrico.test_multicolinealidad(X_post, ['TIIE', 'IGAE', 'FIX', 'Petróleo'])
