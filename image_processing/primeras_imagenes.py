import torch
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path

base_path = Path(__file__).parent
data_path = base_path / "data"
ruta_imagen = data_path / "radiografia_muestra.jpg"

# 1. Cargar la imagen cruda 
# Pillow (PIL) es la librería más limpia para abrir la imagen antes de pasarla a PyTorch
ruta_imagen = "radiografia_muestra.jpg" # Cambia esto por tu imagen real
try:
    img_original = Image.open(ruta_imagen)
except FileNotFoundError:
    print(f"Error: No se encontró la imagen en {ruta_imagen}")
    exit()

# 2. Definir el pipeline de procesamiento (Estilo declarativo y moderno)
# Aquí configuras las reglas. PyTorch se encarga de ejecutarlas de forma ultra-optimizada.
pipeline = v2.Compose([
    v2.Grayscale(num_output_channels=1),     # 1 canal (blanco y negro), elimina el RGB innecesario
    v2.Resize(size=(256, 256)),              # Estandariza la resolución inicial
    v2.CenterCrop(size=(224, 224)),          # Recorta el centro exactamente a 224x224 (estándar en IA)
    v2.ToImage(),                            # Convierte la imagen al formato base de PyTorch
    v2.ToDtype(torch.float32, scale=True)    # Magia pura: Normaliza los píxeles de 0-255 a 0.0-1.0
])

# 3. Ejecutar la transformación
# Pasas la imagen por el pipeline como si fuera una simple función
tensor_final = pipeline(img_original)

# 4. Inspeccionar el resultado (Neo-brutalismo: ver los datos crudos)
print("=== Resultados del Procesamiento ===")
print(f"Estructura del Tensor: {tensor_final.shape}") 
# Salida esperada: torch.Size([1, 224, 224]) -> [Canales, Alto, Ancho]

print(f"Tipo de dato: {tensor_final.dtype}")
print(f"Rango de valores: Min = {tensor_final.min():.4f}, Max = {tensor_final.max():.4f}")

# El 'tensor_final' ya está listo para ser inyectado directamente en una red neuronal.