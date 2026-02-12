import numpy as np
from colorstreak import Logger as log


et = np.array([0.40, 0.35, 0.30, 0.25, 0.20, 0.10, 0.05, 0.00, -0.05, -0.10])
bg_pvalue = 0.003

log.metric("Fórmula DW: Suma de (e_t - e_{t-1})^2 / Suma de (e_t)^2")

numerador = np.sum(np.diff(et)**2)
denominador = np.sum(et**2)
dw = numerador / denominador

log.info(f"Estadístico Durbin-Watson (DW) calculado: {dw:.4f}")



log.info("--- Análisis de Autocorrelación ---")


if dw < 1.0:
    log.debug(f"DW ({dw:.2f}) cercano a 0 -> Indica Autocorrelación POSITIVA.")
elif dw > 3.0:
    log.debug(f"DW ({dw:.2f}) cercano a 4 -> Indica Autocorrelación NEGATIVA.")
else:
    log.debug("DW cercano a 2 -> No hay autocorrelación aparente.")


if bg_pvalue < 0.05:
    log.debug(f"Breusch-Godfrey p-value ({bg_pvalue}) < 0.05 -> Rechazamos H0 (Hay autocorrelación).")

log.error("CONCLUSIÓN FINAL: Existe Autocorrelación Positiva Grave.")

log.info("--- 2 Consecuencias sobre Inferencia ---")
log.debug("1. Estimadores Ineficientes: Los coeficientes (Betas) ya no tienen varianza mínima (no son MELI/BLUE).")
log.debug("2. Inferencia Inválida: Los errores estándar se subestiman, lo que infla la t-statistic y causa falsos positivos.")


log.info("--- 2 Correcciones Propuestas ---")
log.metric("Estructural: Modificar el modelo agregando un rezago de la dependiente (Yt-1) o una tendencia.")
log.metric("Econométrica: Utilizar Errores Estándar Robustos de Newey-West (HAC) para corregir la varianza.")