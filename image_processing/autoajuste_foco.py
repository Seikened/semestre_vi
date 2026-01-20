from colorstreak import Logger as log

def medida(valor: float, unidad_origen: str, unidad_destino: str) -> float:
    unidades = {
        # micrometros
        'um': 0.001,
        # microcentimetros
        'ucm': 0.01,
        # micromilimetros
        'umm': 0.1,
        # milimetros
        'mm': 1, 
        # centimetros
        'cm': 10, 
        # metros
        'm': 1000,
        }

    if unidad_origen not in unidades or unidad_destino not in unidades:
        lista_unidades = [ key for key ,value in unidades.items()]
        mensaje = f"Unidades no soportadas. Use: {lista_unidades}"
        raise ValueError(mensaje)

    valor_mm = valor * unidades[unidad_origen] 
    valor_convertido = valor_mm / unidades[unidad_destino]  

    return valor_convertido





px = 1
pixel_size = 3.45 # micrometers

ancho_c = (2445/px)*pixel_size
alto_c = (1920/px)*pixel_size


ancho_mm = medida(ancho_c, 'um', 'mm') # Convertir a mm
alto_mm = medida(alto_c, 'um', 'mm')   # Convertir a mm


hi = ancho_mm
do = 50 #cm
foco = 8 # mm


def calcular_distancia_imagen(hi, do, foco):
    foco_cm = foco / 10  # Convertir foco a cm
    di = 1 / ((1 / foco_cm) - (1 / do))
    return di





log.metric(f"Ancho del sensor: {ancho_mm:.2f} mm")
log.metric(f"Alto del sensor: {alto_mm:.2f} mm")