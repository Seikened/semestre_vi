from pathlib import Path
from typing import Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from colorstreak import Logger as log
from PIL import Image
from torchvision.transforms import v2
from torchvision.utils import save_image

"""
Módulo de Procesamiento de Imágenes
Contiene la clase VisionNode para manipulación fluida de tensores de imagen.
"""

# ==========================================
# Helpers
# ==========================================
def get_image_path(name: str, base_dir: Path | str | None = None) -> Path:
    """
    Obtiene la ruta a una imagen.

    Args:
        name:     Nombre del archivo (puede incluir subcarpetas, ej. "patrones/x.bmp").
        base_dir: Carpeta base donde buscar. Si es None usa `image_processing/data/`.
    """
    if base_dir is None:
        base = Path(__file__).parent / "data"
    else:
        base = Path(base_dir)
    return base / name


def tag(tipo: str, hace: str = "", depende_de: tuple[str, ...] = ()):
    """
    Decorador de metadata para documentar la API.

    Args:
        tipo:       "factory" | "transformacion" | "grafica" | "utilidad" | "binarizacion"
        hace:       Resumen corto (una línea) de lo que hace.
        depende_de: Otros métodos/funciones de los que depende internamente.

    Uso:
        @tag(tipo="grafica", hace="Renderiza el tensor", depende_de=("tensor",))
        def mostrar(self): ...
    """
    def deco(func):
        func._tag = {"tipo": tipo, "hace": hace, "depende_de": tuple(depende_de)}
        return func
    return deco


# ==========================================
# VisionNode
# ==========================================
class VisionNode:
    """
    Representación inmutable de un tensor de imagen.
    Cada operación devuelve un nuevo VisionNode, permitiendo encadenar métodos (Fluent API).
    """
    def __init__(self, tensor: torch.Tensor, title: str = "Imagen"):
        self.tensor = tensor
        self.title = title
        self.channels, self.height, self.width = self.tensor.shape
        self.is_grayscale = (self.channels == 1)

    # ==========================================
    # OPERADORES ARITMÉTICOS (estilo numpy / MathCad)
    # ==========================================
    # No aplican clamp automáticamente; usa `.clip()` al final si quieres
    # recortar al rango [0,1]. Ejemplo MathCad:
    #     clip(IMAGEN/3 + 127)    ← escala 0-255
    # Equivalente aquí (escala 0-1):
    #     (img/3 + 0.5).clip()

    @staticmethod
    def _obtener_tensor(otro) -> torch.Tensor | float | int:
        """Extrae tensor si es VisionNode; deja pasar escalares."""
        return otro.tensor if isinstance(otro, VisionNode) else otro

    @staticmethod
    def _etiqueta(otro) -> str:
        return otro.title if isinstance(otro, VisionNode) else f"{otro:g}"

    def _combinar(self, otro, op: str, func) -> Self:
        tensor_otro = self._obtener_tensor(otro)
        nuevo_tensor = func(self.tensor, tensor_otro)
        return self.__class__(nuevo_tensor, title=f"({self.title} {op} {self._etiqueta(otro)})")

    def __add__(self, otro) -> Self:      return self._combinar(otro, "+", lambda a, b: a + b)
    def __sub__(self, otro) -> Self:      return self._combinar(otro, "-", lambda a, b: a - b)
    def __mul__(self, otro) -> Self:      return self._combinar(otro, "*", lambda a, b: a * b)
    def __truediv__(self, otro) -> Self:  return self._combinar(otro, "/", lambda a, b: a / b)

    def __radd__(self, otro) -> Self:     return self._combinar(otro, "+", lambda a, b: b + a)
    def __rsub__(self, otro) -> Self:     return self._combinar(otro, "-", lambda a, b: b - a)
    def __rmul__(self, otro) -> Self:     return self._combinar(otro, "*", lambda a, b: b * a)
    def __rtruediv__(self, otro) -> Self: return self._combinar(otro, "/", lambda a, b: b / a)

    def __neg__(self) -> Self:
        return self.__class__(-self.tensor, title=f"(-{self.title})")

    @tag(tipo="utilidad",
         hace="Promedio aritmético entre self y una lista de nodos del mismo shape.",
         depende_de=("tensor",))
    def promediar(self, otros: list["VisionNode"]) -> Self:
        """
        Promedia self con N nodos adicionales (mismo shape):
            salida = (self + Σ otros) / (1 + N)

        Útil para reducir ruido por integración temporal — técnica clásica en
        astronomía y microscopía cuando la señal es estática y el ruido es
        independiente entre tomas.
        """
        if not otros:
            return self.__class__(self.tensor.clone(), title=self.title)

        suma = self.tensor.clone()
        for nodo in otros:
            if nodo.tensor.shape != self.tensor.shape:
                raise ValueError(
                    f"Shape incompatible: {nodo.tensor.shape} vs {self.tensor.shape}")
            suma = suma + nodo.tensor

        promedio = suma / (len(otros) + 1)
        return self.__class__(promedio, title=f"Promedio de {len(otros) + 1} imágenes")

    @tag(tipo="transformacion", hace="Clamp al rango [min,max] (por defecto [0,1]).",
         depende_de=("tensor",))
    def clip(self, min: float = 0.0, max: float = 1.0) -> Self:
        """Recorta los valores al rango [min, max]. Análogo a np.clip / MathCad clip."""
        return self.__class__(self.tensor.clamp(min, max), title=f"clip({self.title})")

    @tag(tipo="utilidad", hace="Valor máximo del tensor (escala 0-1).", depende_de=("tensor",))
    def max(self) -> float:
        """Valor máximo del tensor. Útil para verificar rango después de operaciones."""
        return self.tensor.max().item()

    @tag(tipo="utilidad", hace="Valor mínimo del tensor (escala 0-1).", depende_de=("tensor",))
    def min(self) -> float:
        """Valor mínimo del tensor. Útil para verificar rango después de operaciones."""
        return self.tensor.min().item()

    @classmethod
    def describir_api(cls) -> None:
        """
        Imprime una tabla con la metadata (@tag) de todos los métodos públicos
        de la clase y sus padres.

        Para cada nombre de método se muestra la clase que lo declara primero
        en el MRO. Si esa declaración no tiene @tag pero alguna versión heredada
        sí, se hereda el tag (las override sin @tag suelen ser auxiliares).
        """
        def _resolver_func(attr):
            if isinstance(attr, (classmethod, staticmethod)):
                return attr.__func__
            return attr if callable(attr) else None

        def _buscar_tag(nombre: str) -> dict | None:
            for klass in cls.__mro__:
                attr = vars(klass).get(nombre)
                if attr is None:
                    continue
                func = _resolver_func(attr)
                if func is None:
                    continue
                meta = getattr(func, "_tag", None)
                if meta:
                    return meta
            return None

        vistos: set[str] = set()
        filas = []
        for klass in cls.__mro__:
            if klass is object:
                continue
            for nombre, attr in vars(klass).items():
                if nombre.startswith("_") or nombre in vistos:
                    continue
                if _resolver_func(attr) is None:
                    continue
                vistos.add(nombre)
                meta = _buscar_tag(nombre)
                if meta:
                    filas.append((klass.__name__, nombre, meta["tipo"], meta["hace"],
                                  ", ".join(meta["depende_de"]) or "—"))
                else:
                    filas.append((klass.__name__, nombre, "SIN TAG", "", "—"))

        header = ("Clase", "Método", "Tipo", "Hace", "Depende de")
        filas.sort(key=lambda r: (r[2], r[0], r[1]))
        anchos = [max(len(str(f[i])) for f in filas + [header]) for i in range(5)]

        print(" | ".join(str(h).ljust(anchos[i]) for i, h in enumerate(header)))
        print("-+-".join("-" * a for a in anchos))
        for f in filas:
            print(" | ".join(str(f[i]).ljust(anchos[i]) for i in range(5)))

    @classmethod
    @tag(tipo="factory", hace="Carga imagen desde archivo y la convierte a tensor RGB normalizado [0,1].")
    def desde_archivo(cls, path: Path) -> Self:
        """Factory method para instanciar directamente desde una ruta."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_raw = Image.open(path).convert("RGB")

        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        tensor_rgb = transform(img_raw).to(device)
        return cls(tensor_rgb, title=path.name)

    # ==========================================
    # MÉTODOS DE TRANSFORMACIÓN (Devuelven VisionNode)
    # ==========================================

    @tag(tipo="transformacion", hace="Invierte el tensor (1-x).", depende_de=("tensor",))
    def negativo(self) -> Self:
        """Invierte los valores del tensor matemáticamente."""
        tensor_neg = 1.0 - self.tensor
        return self.__class__(tensor_neg, title=f"Negativo de {self.title}")

    @tag(tipo="transformacion", hace="Colapsa RGB a 1 canal promediando (media aritmética, no luminancia).",
         depende_de=("tensor",))
    def escala_grises(self) -> Self:
        """
        Convierte a blanco y negro colapsando los canales RGB con media aritmética:
            gris = (R + G + B) / 3

        NOTA: Esto NO es la conversión perceptualmente correcta. Librerías estándar
        (OpenCV, scikit-image, Kornia, PIL) usan luminancia ITU-R BT.601:
            gris = 0.299·R + 0.587·G + 0.114·B
        que pondera el verde más alto porque el ojo humano es más sensible a él.

        Aquí se usa la media simple por claridad pedagógica (todos los canales
        contribuyen igual). Si necesitas equivalencia visual con otras libs,
        aplica los pesos BT.601 manualmente.
        """
        tensor_bn = torch.mean(self.tensor, dim=0, keepdim=True)
        return self.__class__(tensor_bn, title=f"B/N de {self.title}")

    @tag(tipo="utilidad", hace="Separa RGB en 3 nodos independientes.", depende_de=("tensor", "is_grayscale"))
    def separar_canales(self) -> dict[str, Self]:
        """Separa la imagen en sus 3 canales base como nodos independientes."""
        if self.is_grayscale:
            raise ValueError("No se pueden separar canales de una imagen en blanco y negro.")

        return {
            "Rojo": self.__class__(self.tensor[0:1, :, :], title="Canal Rojo"),
            "Verde": self.__class__(self.tensor[1:2, :, :], title="Canal Verde"),
            "Azul": self.__class__(self.tensor[2:3, :, :], title="Canal Azul")
        }

    @tag(tipo="transformacion", hace="Saturación HSV por pixel: 0=gris/desteñido, 1=color puro.",
         depende_de=("tensor",))
    def saturacion(self) -> Self:
        """
        Croma (componente S del modelo HSV) como nodo de 1 canal en [0,1]:
            S = (Cmax - Cmin) / Cmax

        Mide qué tan "vivo" es el color: un verde puro da S alto; un blanco o
        gris (color desteñido) da S ≈ 0, sin importar su brillo. Por eso revela
        un velo blanquecino sobre una superficie de color aunque el brillo
        cambie poco — donde el brillo apenas reacciona, la saturación cae.
        """
        if self.is_grayscale:
            raise ValueError("La saturación requiere una imagen RGB.")
        cmax = self.tensor.amax(dim=0, keepdim=True)
        cmin = self.tensor.amin(dim=0, keepdim=True)
        s = (cmax - cmin) / cmax.clamp_min(1e-6)
        return self.__class__(s, title=f"Saturación de {self.title}")

    @tag(tipo="utilidad", hace="Separa la imagen en sus 3 canales HSV (matiz, saturación, valor).",
         depende_de=("tensor", "is_grayscale"))
    def separar_hsv(self) -> dict[str, Self]:
        """
        Convierte RGB→HSV y devuelve los 3 canales como nodos de 1 canal en [0,1]:
            - Matiz (H):      tono, normalizado H/360 (el rojo cae en 0 y en 1).
            - Saturación (S): croma, (Cmax - Cmin) / Cmax.
            - Valor (V):      brillo, Cmax.

        Análogo a separar_canales pero en HSV, donde el color (matiz, saturación)
        queda desacoplado del brillo (valor).
        """
        if self.is_grayscale:
            raise ValueError("No se puede separar HSV de una imagen en blanco y negro.")

        r, g, b = self.tensor[0], self.tensor[1], self.tensor[2]
        cmax, indice_max = self.tensor.max(dim=0)
        cmin = self.tensor.amin(dim=0)
        delta = (cmax - cmin).clamp_min(1e-6)

        sextante = torch.stack([
            ((g - b) / delta) % 6.0,   # Cmax = R
            (b - r) / delta + 2.0,     # Cmax = G
            (r - g) / delta + 4.0,     # Cmax = B
        ])
        matiz = torch.gather(sextante, 0, indice_max.unsqueeze(0)).squeeze(0) / 6.0
        matiz = torch.where(cmax == cmin, torch.zeros_like(matiz), matiz)
        saturacion = (cmax - cmin) / cmax.clamp_min(1e-6)

        return {
            "Matiz":      self.__class__(matiz.unsqueeze(0),       title="Matiz (H)"),
            "Saturación": self.__class__(saturacion.unsqueeze(0),  title="Saturación (S)"),
            "Valor":      self.__class__(cmax.unsqueeze(0),        title="Valor (V)"),
        }

    @tag(tipo="transformacion", hace="Multiplica por factor con clamp [0,1].", depende_de=("tensor",))
    def ganancia(self, factor: float) -> Self:
        """Aplica un factor de multiplicación al tensor forzando los límites [0, 1]."""
        tensor_mult = torch.clamp(self.tensor * factor, min=0.0, max=1.0)
        return self.__class__(tensor_mult, title=f"x{factor} de {self.title}")

    @tag(tipo="transformacion", hace="Normalización min-max al rango [0,1].", depende_de=("tensor",))
    def estirar_contraste(self) -> Self:
        """
        Expande el rango de píxeles para que ocupen todo el espectro [0.0, 1.0].
        Ideal cuando los valores de los píxeles están muy agrupados (poco contraste).
        También conocido como Normalización Min-Max.
        """
        min_val = torch.min(self.tensor)
        max_val = torch.max(self.tensor)

        # Prevenir división por cero si la imagen es de un solo color
        if max_val == min_val:
            return self.__class__(self.tensor.clone(), title=f"Estirado de {self.title}")

        tensor_stretched = (self.tensor - min_val) / (max_val - min_val)
        return self.__class__(tensor_stretched, title=f"Estirado de {self.title}")

    # ==========================================
    # MÉTODOS DE BINARIZACIÓN (Estrategias)
    # ==========================================

    @tag(tipo="binarizacion", hace="Umbral global: tensor > th.", depende_de=("tensor",))
    def binarizar(self, threshold: float = 0.5) -> Self:
        """MODO BRUTO (Global Threshold). De 0 a 1, el umbral se aplica directamente al tensor."""
        tensor_bin = (self.tensor > threshold).float()
        return self.__class__(tensor_bin, title=f"Bin (th={threshold}) de {self.title}")

    @tag(tipo="binarizacion", hace="Umbral por rango: lower ≤ x ≤ upper.", depende_de=("tensor",))
    def binarizar_rango(self, lower: float, upper: float) -> Self:
        """MODO RANGO (Bandpass / InRange)."""
        tensor_bin = ((self.tensor >= lower) & (self.tensor <= upper)).float()
        return self.__class__(tensor_bin, title=f"Bin Rango [{lower}-{upper}] de {self.title}")

    @tag(tipo="binarizacion", hace="Umbral adaptativo por media local vía integral image (O(1) por pixel).",
         depende_de=("tensor", "torch.cumsum"))
    def binarizar_adaptativo(self, kernel_size: int = 15, c: float = 0.05) -> Self:
        """MODO INDUSTRIA (Adaptive Thresholding con Kernel).
        Implementación con integral image (summed-area table): O(1) por pixel,
        independiente del tamaño del kernel.
        """
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size debe ser impar, recibido {kernel_size}")

        ks = kernel_size
        pad = ks // 2
        C, H, W = self.tensor.shape

        padded = F.pad(self.tensor, (pad, pad, pad, pad), mode="reflect")
        integral = F.pad(padded.cumsum(-1).cumsum(-2), (1, 0, 1, 0))

        suma = (integral[:, ks:ks + H, ks:ks + W]
                - integral[:, :H,        ks:ks + W]
                - integral[:, ks:ks + H, :W]
                + integral[:, :H,        :W])
        local_mean = suma / (ks * ks)
        tensor_bin = (self.tensor > (local_mean - c)).float()

        return self.__class__(tensor_bin, title=f"Bin Adaptativo (k={kernel_size}) de {self.title}")

    # ==========================================
    # MÉTODOS AVANZADOS
    # ==========================================

    @tag(tipo="transformacion", hace="Log transform: aclara zonas oscuras.", depende_de=("tensor",))
    def transformacion_log(self) -> Self:
        """Transformación logarítmica. Aclara zonas oscuras."""
        device = self.tensor.device
        tensor_255 = self.tensor * 255.0
        denominador = torch.log(torch.tensor(256.0, device=device))
        tensor_log = torch.log1p(tensor_255) / denominador
        return self.__class__(tensor_log, title=f"Log Transform de {self.title}")

    @tag(tipo="transformacion", hace="Gamma correction: x^gamma.", depende_de=("tensor",))
    def transformacion_gamma(self, gamma: float) -> Self:
        """Transformación exponencial (Gamma Correction)."""
        tensor_gamma = torch.pow(self.tensor, gamma)
        return self.__class__(tensor_gamma, title=f"Gamma (g={gamma}) de {self.title}")

    @tag(tipo="transformacion", hace="Reduce profundidad de bits por canal.", depende_de=("tensor",))
    def cuantizar(self, n_bits: int = 8) -> Self:
        """Cuantización reduciendo el número de bits por canal."""
        levels = 2 ** n_bits
        tensor_quant = torch.floor(self.tensor * (levels - 1)) / (levels - 1)
        return self.__class__(tensor_quant, title=f"Cuantizado ({n_bits} bits) de {self.title}")

    @tag(tipo="transformacion", hace="Pseudo-color térmico vía colormap.", depende_de=("escala_grises",))
    def pseudocolor_infrarrojo(self, cmap_name: str = "inferno") -> Self:
        """Aplica un mapa de color simulando una cámara infrarroja (térmica)."""
        base = self if self.is_grayscale else self.escala_grises()
        base_np = base.tensor.squeeze(0).cpu().numpy()

        cmap = mpl.colormaps[cmap_name]
        colored_rgba = cmap(base_np)
        tensor_rgb = torch.tensor(
            colored_rgba[..., :3],
            dtype=torch.float32,
            device=self.tensor.device
        ).permute(2, 0, 1)

        return self.__class__(tensor_rgb, title=f"Infrarrojo Térmico ({cmap_name}) de {self.title}")

    @tag(tipo="transformacion", hace="Permuta canales estilo Kodak Aerochrome.", depende_de=("tensor",))
    def falso_color_infrarrojo(self) -> Self:
        """Simula película Infrarroja a color (estilo Kodak Aerochrome)."""
        if self.is_grayscale:
            raise ValueError("El infrarrojo falso color requiere una imagen RGB.")

        r = self.tensor[0:1, :, :]
        g = self.tensor[1:2, :, :]
        b = self.tensor[2:3, :, :]

        ir_tensor = torch.cat([g, r, b], dim=0)
        return self.__class__(ir_tensor, title=f"Infrarrojo Aerochrome de {self.title}")

    # ==========================================
    # MÉTODOS DE VISUALIZACIÓN
    # ==========================================

    @tag(tipo="grafica", hace="Mapa de diferencia absoluta entre dos nodos.", depende_de=("mostrar",))
    def mostrar_diferencias(self, compare_to: "VisionNode", magnifier: float = 1.0, block: bool = True) -> Self:
        """Calcula y muestra el mapa de error absoluto entre dos nodos."""
        if self.tensor.shape != compare_to.tensor.shape:
            raise ValueError(f"Incompatibilidad de dimensiones: {self.tensor.shape} vs {compare_to.tensor.shape}")

        tensor_diff = torch.abs(self.tensor - compare_to.tensor)
        tensor_diff = torch.clamp(tensor_diff * magnifier, min=0.0, max=1.0)
        title = f"Diferencias: {self.title} vs {compare_to.title}"
        return self.__class__(tensor_diff, title=title).mostrar(block=block)

    @tag(tipo="grafica", hace="Renderiza el tensor como imagen.", depende_de=("tensor", "is_grayscale"))
    def mostrar(self, block: bool = True) -> Self:
        """Renderiza el nodo actual."""
        img_disp = self.tensor.permute(1, 2, 0).cpu().squeeze().numpy()
        cmap = "gray" if self.is_grayscale else None

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img_disp, cmap=cmap, vmin=0, vmax=1)
        plt.title(self.title)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=block)
        return self

    @tag(tipo="grafica", hace="Histograma por canal con escala log auto.", depende_de=("tensor", "is_grayscale"))
    def histograma(self, block: bool = True) -> Self:
        """Calcula y muestra el histograma del nodo en escala logarítmica.

        Layout:
        - Color:    cuadrícula 2x2 → imagen | R
                                             G   | B
        - Escala de grises: cuadrícula 1x2 → imagen | B/N
        """
        img_disp = self.tensor.permute(1, 2, 0).cpu().squeeze().numpy()
        cmap_img = "gray" if self.is_grayscale else None

        if self.is_grayscale:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            ax_img = axes[0]
            hist_axes = [axes[1]]
            canales = [("B/N", "black", self.tensor[0])]
        else:
            fig = plt.figure(figsize=(12, 8))
            ax_img  = fig.add_subplot(2, 2, 1)
            ax_r    = fig.add_subplot(2, 2, 2)
            ax_g    = fig.add_subplot(2, 2, 3)
            ax_b    = fig.add_subplot(2, 2, 4)
            hist_axes = [ax_r, ax_g, ax_b]
            canales = [
                ("Rojo",  "red",   self.tensor[0]),
                ("Verde", "green", self.tensor[1]),
                ("Azul",  "blue",  self.tensor[2]),
            ]

        # Imagen en el primer cuadrante
        ax_img.imshow(img_disp, cmap=cmap_img, vmin=0, vmax=1)
        ax_img.set_title(self.title, fontsize=11)
        ax_img.axis("off")

        # Histogramas
        for ax, (nombre, color, tensor_canal) in zip(hist_axes, canales):
            datos_255 = tensor_canal.cpu().numpy().flatten() * 255.0
            counts, bin_edges = np.histogram(datos_255, bins=256, range=(0, 256))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Detectar imagen "dispersa" (ej: binarizada): bins no vacíos < 10% del total
            bins_con_datos = np.count_nonzero(counts)
            es_disperso = bins_con_datos < 26  # menos de ~10% de los 256 bins

            ax.bar(bin_centers, counts, width=1.0, color=color, alpha=0.7)
            ax.axvline(x=255, color='red', linestyle='--', alpha=0.6, label='Saturación')

            if not es_disperso:
                ax.set_yscale('log', nonpositive='clip')
                ax.set_ylabel("Píxeles (Log)")
            else:
                ax.set_ylabel("Píxeles")

            ax.set_title(f"Canal {nombre}", fontsize=11)
            ax.set_xlim(-5, 260)
            ax.set_xlabel("Intensidad [0 - 255]")
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc="upper right")

        fig.suptitle(f"Histograma: {self.title}", fontsize=13, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=block)
        return self

    @tag(tipo="utilidad", hace="Guarda el tensor como archivo de imagen (PNG/JPG).",
         depende_de=("tensor",))
    def guardar(self, ruta: Path | str) -> Self:
        """Exporta el tensor [0,1] a disco como imagen. Crea la carpeta si falta."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        save_image(self.tensor, ruta)
        return self

    @tag(tipo="grafica", hace="Grid comparativo: original + canales + B/N.",
         depende_de=("separar_canales", "escala_grises"))
    def mostrar_reporte(self, block: bool = True) -> Self:
        """Renderiza un reporte comparativo entre la imagen original y sus canales."""
        if self.is_grayscale:
            log.warning("El reporte comparativo requiere una imagen RGB.")
            return self

        canales = self.separar_canales()
        bn_node = self.escala_grises()

        nodos_a_graficar = [self] + list(canales.values()) + [bn_node]
        fig, axes = plt.subplots(1, len(nodos_a_graficar), figsize=(20, 5))

        for ax, nodo in zip(axes, nodos_a_graficar):
            img_disp = nodo.tensor.permute(1, 2, 0).cpu().squeeze().numpy()
            cmap = "gray" if nodo.is_grayscale else None

            ax.imshow(img_disp, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(nodo.title)
            ax.axis("off")

        plt.tight_layout()
        plt.show(block=block)
        return self


# ==========================================
# Pruebas / Demo
# ==========================================
def demo_pipeline():
    """Ejecuta un pequeño demo usando imágenes de muestra si están disponibles."""
    fotos = {
        "torax": get_image_path("toraxP2.bmp"),
        "arco": get_image_path("arco1.bmp"),
        "tumba": get_image_path("nd5.bmp"),
        "fondo": get_image_path("fondo_negro.jpeg"),
        "ventana": get_image_path("louvre4.bmp"),
        "taller": get_image_path("taller1.jpg"),
        "infra": get_image_path("infrarrojo/arteriesMIR.jpg"),
    }

    mi_imagen_path = fotos.get("infra")
    if not mi_imagen_path or not mi_imagen_path.exists():
        log.warning("No se encontró la imagen de prueba para el demo.")
        return

    img_original = VisionNode.desde_archivo(mi_imagen_path)
    img_original.mostrar(block=False)

    img_expo = img_original.transformacion_gamma(0.5)
    img_expo.mostrar(block=False)

    img_estirada = img_original.estirar_contraste()
    img_estirada.mostrar(block=False)

    img_ir_termica = img_original.pseudocolor_infrarrojo()
    img_ir_termica.mostrar(block=False)

    log.info("Demo finalizado. Esperando a que se cierren las ventanas...")
    plt.show()

if __name__ == "__main__":
    demo_pipeline()
