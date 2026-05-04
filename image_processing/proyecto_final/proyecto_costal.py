import sys
from pathlib import Path

aqui = Path(__file__).resolve().parent
project_root = aqui.parents[1]
for p in (str(project_root), str(aqui)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt  # noqa: E402
from colorstreak import Logger as log  # noqa: E402

from image_processing import DerivativeVisionNode  # noqa: E402

from notch import aplicar_cruz, mostrar_mascara_cruz  # noqa: E402


"""
Proyecto: Eliminación del patrón de tejido en una imagen de costal
------------------------------------------------------------------
La imagen 'saco_doble_lampara.bmp' tiene un patrón cuasi-periódico:
la trama del polipropileno tejido. Diagnóstico inicial reveló que la
trama es PREDOMINANTEMENTE VERTICAL — todos los picos del espectro
están sobre el eje horizontal del espectro (v ≈ 0), en una secuencia
armónica desde u≈49 hasta u≈600+.

Por eso usamos un filtro de BANDA HORIZONTAL solamente (sigma_v=None):
matamos toda la línea v=0 del espectro (excepto la zona DC), y dejamos
intacta la banda vertical donde vive la información del texto y el
escudo. Si el tejido fuera cuadrado pondríamos también sigma_v para
tener cruz completa.

Pipeline:
  A) Visualización del patrón:
     1. Imagen original en grises.
     2. Señal 1D + FFT por fila (slider).
     3. Espectro 2D con picos detectados (vista rápida).
     5. Joint plot log: ves la fila de picos sobre el eje horizontal.

  B) Filtrado de banda horizontal:
     6. Mostrar la máscara H(u,v) — banda v≈0 con DC libre.
     7. Aplicar el filtro (FFT → ×H → IFFT).
     8. Imagen filtrada.
     9. Joint plot DESPUÉS — el eje horizontal debería estar oscuro.
    10. Diagnóstico: diferencia |original - filtrada|.
"""


IMAGEN = aqui / "saco_doble_lampara.bmp"

# Detección de picos (solo se usa para visualización en pasos 3 y 5)
N_PICOS = 8
EXCLUIR_RADIO_DC = 30
VENTANA_SUPRESION = 15

# Parámetros del filtro de banda
#   sigma_h: anchura (px) de la banda horizontal apagada (mata rayas verticales).
#   sigma_v: anchura (px) de la banda vertical apagada (mata rayas horizontales).
#            Si es None, no se filtra esa banda.
#   radio_dc_libre: radio (px) alrededor del DC preservado.
# Para este costal (trama vertical): sigma_h alrededor de 10-15, sigma_v=None.
# Si tuvieras un tejido cuadrado, pondrías sigma_v también (≈ sigma_h).
SIGMA_H = 10.0
SIGMA_V = None
RADIO_DC_LIBRE = 30.0


def main() -> None:
    if not IMAGEN.exists():
        log.error(f"No se encontró la imagen: {IMAGEN}")
        return

    log.step("1. Cargando imagen y convirtiendo a grises")
    img_gris = DerivativeVisionNode.desde_archivo(IMAGEN).escala_grises()
    img_gris.title = "Saco — original"
    #img_gris.mostrar(block=False)

    log.step("2. Señal 1D + FFT por fila (slider) — primera vista periodicidad")
    img_gris.senal_por_canal(block=False)

    log.step("3. Espectro 2D con picos detectados (vista rápida)")
    img_gris.espectro_2d_picos(
        n_picos=N_PICOS,
        excluir_radio_dc=EXCLUIR_RADIO_DC,
        ventana_supresion=VENTANA_SUPRESION,
        block=False,
    )

    log.step("5. Mismo joint plot en escala log (para ver toda la distribución)")
    img_gris.espectro_2d_con_perfiles(
        n_picos=N_PICOS,
        excluir_radio_dc=EXCLUIR_RADIO_DC,
        ventana_supresion=VENTANA_SUPRESION,
        escala_perfiles="log",
        block=False,
    )

    # ── B) Filtrado de banda horizontal ──────────────────────
    canal = img_gris.tensor[0].cpu().numpy()

    log.step(f"6. Mostrando máscara H(u,v) — σh={SIGMA_H}, σv={SIGMA_V}, "
             f"DC libre={RADIO_DC_LIBRE}")
    mostrar_mascara_cruz(canal.shape, SIGMA_H, SIGMA_V, RADIO_DC_LIBRE, block=False)

    log.step("7. Aplicando filtro (FFT → ×H → IFFT)")
    img_limpia = aplicar_cruz(img_gris, sigma_h=SIGMA_H, sigma_v=SIGMA_V,
                               radio_dc_libre=RADIO_DC_LIBRE)

    log.step("8. Imagen filtrada")
    img_limpia.mostrar(block=False)

    log.step("9. Joint plot DESPUÉS — el eje horizontal del espectro debería estar oscuro")
    img_limpia.espectro_2d_con_perfiles(
        n_picos=N_PICOS,
        excluir_radio_dc=EXCLUIR_RADIO_DC,
        ventana_supresion=VENTANA_SUPRESION,
        escala_perfiles="lineal",
        block=False,
    )

    log.step("10. Diagnóstico: mapa de diferencias |original - filtrada|")
    img_gris.mostrar_diferencias(img_limpia, magnifier=5.0, block=False)

    log.info("Cierra las ventanas para finalizar.")
    plt.show()


if __name__ == "__main__":
    main()
