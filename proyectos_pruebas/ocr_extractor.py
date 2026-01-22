import cv2
import easyocr
import numpy as np
from pathlib import Path
from colorstreak import Logger as log


DATA_DIR = Path(__file__).resolve().parent / "data"
imagen_path = DATA_DIR / "tu_documento_medico.jpeg"
existe = imagen_path.exists()



# --- CONFIGURACIÓN ---
# Reemplaza esto con la ruta de tu imagen (foto de celular, escaneo, etc.)
RUTA_IMAGEN = str(imagen_path)  
RUTA_SALIDA = DATA_DIR  / "resultado_visual.jpeg"

# Idiomas a buscar (añade 'en' si hay términos médicos en inglés)
IDIOMAS = ['es'] 

# Usa gpu=True si tienes CUDA configurado, si no, usa False (más lento)
USAR_GPU = False 
# ---------------------


def visualizar_ocr(ruta_img, idiomas, gpu_activada):
    print(f"Cargando imagen: {ruta_img}...")
    # 1. Cargar la imagen con OpenCV
    img = cv2.imread(ruta_img)
    if img is None:
        print("Error: No se pudo cargar la imagen. Revisa la ruta.")
        return

    # Hacemos una copia para dibujar sobre ella y no dañar la original en memoria
    img_resultado = img.copy()

    print("Iniciando motor OCR (esto puede tardar un poco la primera vez)...")
    # 2. Inicializar EasyOCR
    reader = easyocr.Reader(idiomas, gpu=gpu_activada)

    print("Detectando texto y coordenadas...")
    # 3. Ejecutar la lectura.
    # detail=1 (por defecto) devuelve las coordenadas, el texto y la confianza.
    resultados = reader.readtext(img)

    print(f"Se encontraron {len(resultados)} bloques de texto. Dibujando...")

    # 4. Iterar sobre los resultados y dibujar
    # El formato de 'res' es: ( [coordenadas_caja], "texto detectado", confianza )
    for (bbox, texto, confianza) in resultados:
        # EasyOCR devuelve las coordenadas de las 4 esquinas de la caja.
        # bbox = [[x_izq_arr, y_izq_arr], [x_der_arr, y_der_arr], [x_der_aba, y_der_aba], [x_izq_aba, y_izq_aba]]
        
        # Para dibujar un rectángulo simple con OpenCV, necesitamos la esquina superior izquierda y la inferior derecha.
        # Convertimos a enteros porque los píxeles no tienen decimales.
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # --- DIBUJAR LA CAJA (Rectángulo) ---
        # Color Verde (BGR): (0, 255, 0), Grosor: 2px
        cv2.rectangle(img_resultado, top_left, bottom_right, (0, 255, 0), 2)

        # --- DIBUJAR EL TEXTO DETECTADO ---
        # Ponemos el texto un poco más arriba de la caja
        posicion_texto = (top_left[0], top_left[1] - 10)
        
        # Usamos una fuente simple. 
        # Color Rojo (BGR): (0, 0, 255), Escala: 0.6, Grosor: 2
        cv2.putText(img_resultado, texto, posicion_texto, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 5. Guardar el resultado
    cv2.imwrite(RUTA_SALIDA, img_resultado)
    print(f"¡Listo! Imagen guardada como: {RUTA_SALIDA}")
    print("Abre esa imagen para ver qué detectó el modelo.")

if __name__ == "__main__":
    # Crear un archivo de prueba dummy si no existe para que el script corra la primera vez
    import os
    if not os.path.exists(RUTA_IMAGEN):
        print(f"ATENCION: No existe {RUTA_IMAGEN}. Por favor coloca una imagen real.")
        # Creamos una imagen negra de prueba con texto
        dummy = np.zeros((500, 500, 3), np.uint8)
        cv2.putText(dummy, 'PRUEBA DE EXPEDIENTE 123', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite(RUTA_IMAGEN, dummy)
        print("Se ha creado una imagen de prueba automática.")
    
    visualizar_ocr(RUTA_IMAGEN, IDIOMAS, USAR_GPU)