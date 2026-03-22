import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from image_processing.ajustes_dinamicos import DynamicVisionNode

DATA_DIR = BASE_DIR / "image_processing" / "practica" / "data"

files = {
    "korda":       "korda.jpg",
    "torax":       "toraxP.bmp",
    "laboratorio": "laboratorio.bmp",
    "grafica":     "GRAFICA2.bmp",
    "tumba":       "tumba.bmp",
    "austin":      "austincolor.jpg",
}


def load_image(nombre:str) -> DynamicVisionNode:
    return DynamicVisionNode.desde_archivo(DATA_DIR / files[nombre])


# Ejercicio 1
def ejercicio_uno_a():
    """
    a) Una radiografía fue tomada para diagnosticar en un paciente problemas en la columna vertebral. Por
    un error del operador de la máquina de rayos X, los huesos no aparecen de manera clara. Como el
    paciente no puede ser expuesto nuevamente, debido a que ya se ha tomado demasiadas radiografías
    para un año, se le pide a usted que procese dicha radiografía para mejorar la definición de los huesos.
    Explique cómo seleccionó el procesamiento aplicado.
    """
    imagen: DynamicVisionNode = load_image("torax")
    #imagen.histograma(block=False)
    imagen.mostrar(block=False)

    img_ganancia = imagen.ganancia(1.2)

    # Solo gama
    img_tranform = img_ganancia.transformacion_gamma(2.0)
    img_tranform.mostrar(block=False)


def ejercicio_uno_b():
    """
    b) Una vez que ha regresado el paciente con su radiografía al consultorio, el médico sospecha de
    problemas en el pulmón y el hígado al enterarse que el paciente consume alcohol y tabaco en
    cantidades no recomendables. ¿Podría usted mejorar la radiografía para facilitar el diagnóstico en este
    caso? Explique cómo seleccionó el procesamiento aplicado.
    """
    imagen: DynamicVisionNode = load_image("torax")
    imagen.mostrar(block=False)
    
    
    img_negativo = (
        imagen
        .escala_grises()
        .negativo()
        .transformacion_gamma(2.0))
    img_negativo.mostrar(block=False)


def ejercicio_uno_c():
    """
    c) Después de discutir sus malos hábitos, el paciente comenta de cierto dolor en la articulación de la
    cadera, justo en la parte alta de la pierna derecha. Mejore la imagen de tal forma que sea mas clara esta
    articulación. Explique cómo seleccionó el procesamiento aplicado.
    """
    imagen: DynamicVisionNode = load_image("torax")
    #imagen.histograma(block=False)
    imagen.mostrar(block=False)

    img_ganancia = imagen.ganancia(1.2)

    # Solo gama
    img_tranform = img_ganancia.transformacion_gamma(2.0)
    #img_tranform.histograma(block=False)
    img_tranform.mostrar(block=False)

    entrada = (0, 149, 150, 255)
    salida  = (0,  50,  50, 255)

    img_linear = img_ganancia.transformacion_lineal(entrada=entrada, salida=salida)
    #img_linear.histograma(block=False)
    img_linear.mostrar(block=False)

    # Gama + linear
    gama_mas_linear = (
        img_ganancia
        .transformacion_gamma(1.3)
        .transformacion_lineal(entrada=entrada, salida=salida)
    )
    #gama_mas_linear.histograma(block=False)
    gama_mas_linear.mostrar(block=False)

# Ejercicio 2
def ejercicio_dos():
    """
    La imagen laboratorio.bmp fue tomada usando una cámara digital industrial (Genie Nano de Dalsa
    Teledyne). Sin embargo, la apariencia no es agradable (regiones muy obscuras) y no se parece a la
    fotografía que obtendría con un teléfono celular. Explique por que esta fotografía no se ve como una
    fotografía de cámara de consumidor. Arregle la fotografía para obtener una mejor apariencia.
    """

    imagen: DynamicVisionNode = load_image("laboratorio")
    imagen.histograma(block=False)

    
    
    img_tranform = (
        imagen
        .transformacion_gamma(.5)
    )
    img_tranform.histograma(block=False)

    img_acomuladp = (
        imagen
        .transformacion_gamma(0.5)
        .ecualizar()
    )
    img_acomuladp.histograma()


# Ejercicio 3
def ejercicio_tres():
    """
    Su profesor de Procesamiento Digital de Imágenes presume que él mismo ha dibujado las gráficas que
    presentó en la clase introductoria (grafica2.bmp). Usted sospecha que en realidad se trata de un plagio
    del libro de texto (Capítulo 4.2.1, Algunas transformaciones de intensidad simples, negativos de
    imágenes). Pruebe de manera convincente su hipótesis.
    """
    pass

# Ejercicio 4
def ejercicio_cuatro():
    """
    Tenemos la imagen tumba.bmp donde el mármol aparece muy claro y el fondo demasiado obscuro.
    Pruebe si puede arreglar esta imagen aplicando simplemente ganancia digital. ¿Qué resultados
    obtiene? Explique usando el histograma.
    """
    imagen: DynamicVisionNode = load_image("tumba")
    imagen.histograma(block=False)

    img_ganancia = imagen.ganancia(2.0)
    img_ganancia.histograma(block=False)

# Ejercicio 5
def ejercicio_cinco():
    """
    Considere la misma imagen tumba.bmp. Esta es una imagen a color, pero comenzaremos con
    procesamiento en blanco y negro. Queremos mejor visibilidad en las zonas obscuras y mejor contraste
    en las zonas de mármol blanco. Defina a mano una función de transformación T para mejorar esta
    imagen. Explique el criterio para definir esta función. Después pruebe usando ajuste de gama y busque
    si puede lograr un mejor resultado. Finalmente intente con ecualización de imagen. Reporte y compare
    las tres funciones de transformación. ¿Cuáles son las diferencias que encontró entre las funciones y sus
    resultados?
    """
    imagen: DynamicVisionNode = load_image("tumba")
    imagen.histograma(block=False)
    
    
    # Transformación Lineal    
    entrada = (0, 40, 70, 255)
    salida  = (0,  140,  180, 255)

    img_linear = (
        imagen
        .transformacion_lineal(entrada=entrada, salida=salida, show_transform=True)
    )
    img_linear.histograma(block=False)
    
    
    # Transformación Gamma
    img_gamma = (
        imagen
        .transformacion_gamma(0.5)
    )
    img_gamma.histograma(block=False)

    #  Ecualización
    img_ecualizacion = (
        imagen
        .ecualizar()
    )
    img_ecualizacion.histograma(block=False)

# Ejercicio 6
def ejercicio_seis():
    """
    Repita el proceso anterior, pero procesando en color. ¿Qué resultados obtuvo? ¿Hay cambios
    significativos en el color aparente? ¿Cómo es diferente el procesamiento en color al procesamiento en
    blanco y negro?
    """
    
    pass


# Ejercicio 7
def ejercicio_siete():
    """
    Genere los vectores de remapeo Tr, Tg y Tb para falso color, tales que una imagen en infrarrojo
    presente la apariencia de la fotografía AUSTINCOLOR.JPG. Reporte cómo llegó a su definición de las
    funciones de transformación Tr, Tg y Tb.
    """
    
    pass

# Ejercicio 8
def ejercicio_ocho():
    """
    Tiene usted la famosa fotografía de Ernesto Guevara tomada por Korda. Efectúe el procesamiento de
    binarización para obtener una imagen en blanco y negro que será impresa en camisetas utilizando
    serigrafía. La serigrafía no permite tonos de grises, sino pixeles blancos o negros (0 y 255). El impresor
    recomienda que la imagen debe contar con una proporción 50% blanco – 50% negro, pero finalmente
    esta decisión depende de usted. Con base en el histograma y los algoritmos que usted conoce, procese
    tal fotografía. Explique los criterios utilizados.
    """
    
    pass


if __name__ == "__main__":
    #ejercicio_uno_a() #✅
    #ejercicio_uno_b() #✅
    #ejercicio_uno_c() #✅
    #ejercicio_dos() #✅
    #ejercicio_tres()
    #ejercicio_cuatro() #✅
    ejercicio_cinco()
    #ejercicio_seis()
    #ejercicio_siete()
    #ejercicio_ocho()
    plt.show()

