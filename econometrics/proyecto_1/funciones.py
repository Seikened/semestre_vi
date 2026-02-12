import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

class AnalisisEconometrico:
    """
    Clase utilitaria para realizar pruebas econométricas estándar:
    - Regresión OLS
    - Durbin-Watson (Autocorrelación)
    - Breusch-Pagan (Heterocedasticidad)
    - VIF (Multicolinealidad)
    """

    @staticmethod
    def estimar_mco(X: np.ndarray, y: np.ndarray, feature_names: list = None, verbose: bool = False):
        """
        Ejecuta una regresión por Mínimos Cuadrados Ordinarios (MCO/OLS).
        Retorna el modelo ajustado y la matriz X con constante añadida.
        """
        # Añadir constante (intercepto) explícitamente, statsmodels no lo hace solo
        X_con_constante = sm.add_constant(X)
        modelo = sm.OLS(y, X_con_constante).fit()
        
        # Imprimir resumen académico
        if verbose:
            print(modelo.summary(xname=['Intercepto'] + feature_names if feature_names else None))
        return modelo, X_con_constante

    @staticmethod
    def test_autocorrelacion(residuals: np.ndarray):
        """
        Prueba de Durbin-Watson.
        Rango: 0 a 4.
        - ~2.0: No hay autocorrelación (Ideal).
        - < 1.5: Posible autocorrelación positiva.
        - > 2.5: Posible autocorrelación negativa.
        """
        dw_stat = durbin_watson(residuals)
        
        estado = "✅ Sin problema grave"
        if dw_stat < 1.5: estado = "⚠️ Alerta: Autocorrelación Positiva"
        elif dw_stat > 2.5: estado = "⚠️ Alerta: Autocorrelación Negativa"
            
        print(f"\n[Autocorrelación] Durbin-Watson: {dw_stat:.3f} -> {estado}")
        return dw_stat

    @staticmethod
    def test_heterocedasticidad(residuals: np.ndarray, X_con_constante: np.ndarray):
        """
        Prueba de Breusch-Pagan.
        H0: Homocedasticidad (Varianza constante) -> P-value > 0.05
        H1: Heterocedasticidad (Problema) -> P-value < 0.05
        """
        lm, p_bp, f_val, p_f = het_breuschpagan(residuals, X_con_constante)
        
        estado = "✅ Homocedasticidad (Varianza estable)"
        if p_bp < 0.05:
            estado = "⚠️ Heterocedasticidad detectada (Varianza inestable)"
            
        print(f"[Heterocedasticidad] Breusch-Pagan p-value: {p_bp:.4f} -> {estado}")
        return p_bp

    @staticmethod
    def test_multicolinealidad(X: np.ndarray, feature_names: list):
        """
        Calcula el VIF (Variance Inflation Factor).
        - VIF > 5 o 10 indica alta multicolinealidad.
        """
        print("\n[Multicolinealidad] Factor de Inflación de Varianza (VIF):")
        
        if X.shape[1] < 2:
            print("  -> No aplica (Se requiere más de 1 variable independiente)")
            return
            
        # Añadir constante para cálculo correcto
        X_const = sm.add_constant(X)
        
        # Iterar desde 1 para saltar la constante (intercepto)
        for i in range(1, X_const.shape[1]):
            val = variance_inflation_factor(X_const, i)
            name = feature_names[i-1] if feature_names else f"Var {i}"
            
            alerta = "⚠️ ALTO" if val > 5 else "✅ OK"
            print(f"  - {name}: {val:.2f} ({alerta})")

    @staticmethod
    def graficar_residuos(y_pred, residuals):
        plt.figure(figsize=(10, 5))
        plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='w')
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.title('Diagnóstico Visual: Residuos vs Predicción')
        plt.xlabel('Valores Predichos')
        plt.ylabel('Residuos')
        plt.grid(True, alpha=0.3)
        plt.show()