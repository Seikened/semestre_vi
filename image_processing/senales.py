import os
import sys

# Asegurar que el root del proyecto esté en sys.path ANTES de importar el paquete
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataclasses import dataclass
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from colorstreak import Logger as log
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

from image_processing.ajustes_dinamicos import DynamicVisionNode
from image_processing.vision_node import get_image_path, tag


"""
Módulo: Análisis Frecuencial de Imágenes (Señales 2D)
-----------------------------------------------------
Una imagen es una señal 2D. Cada fila de pixeles es una señal 1D,
y la imagen completa se puede descomponer en senoidales 2D con la DFT 2D.

Temas:
1. Visualización de señales por canal — ver los valores de intensidad
   de cada canal (R, G, B o gris) como señal 1D para una fila dada.
2. DFT 2D — Transformada Discreta de Fourier en 2 dimensiones (WxH).
3. Kernels y Convolución 2D — multiplicación + suma sobre vecindades.
"""


# ==========================================
# Constantes de visualización
# ==========================================
COLORES_LINEA_RGB = (("Rojo", "red", 0), ("Verde", "green", 1), ("Azul", "blue", 2))
COLORES_LINEA_BN  = (("B/N", "black", 0),)
COLORMAPS_DFT_RGB = (("Rojo", "Reds", 0), ("Verde", "Greens", 1), ("Azul", "Blues", 2))
COLORMAPS_DFT_BN  = (("B/N", "gray", 0),)

COLOR_FILA_REFERENCIA = "yellow"
COLOR_LUPA = "yellow"
INTENSIDAD_MAX = 255
SIGMA_AUTO_BASE = 0.8
TOLERANCIA_KERNEL_SUMA = 1e-3


@dataclass(frozen=True)
class CanalEspec:
    """Describe un canal para iteración: nombre humano, color/cmap, índice en tensor."""
    nombre: str
    color: str
    indice: int


# ==========================================
# SignalVisionNode
# ==========================================
class SignalVisionNode(DynamicVisionNode):
    """
    Hereda de DynamicVisionNode para añadir análisis frecuencial
    y convoluciones, manteniendo el patrón Fluent API.
    """

    # ──────────────────────────────────────────────────────────
    # Helpers internos
    # ──────────────────────────────────────────────────────────

    def _canales_para_lineas(self) -> list[CanalEspec]:
        """Devuelve la spec de canales para gráficas de línea (señal/FFT)."""
        spec = COLORES_LINEA_BN if self.is_grayscale else COLORES_LINEA_RGB
        return [CanalEspec(*c) for c in spec]

    def _canales_para_dft(self) -> list[CanalEspec]:
        """Devuelve la spec de canales para gráficas de imagen DFT (cmaps)."""
        spec = COLORMAPS_DFT_BN if self.is_grayscale else COLORMAPS_DFT_RGB
        return [CanalEspec(*c) for c in spec]

    def _imagen_para_imshow(self) -> np.ndarray:
        """Convierte el tensor a array HxW(xC) listo para matplotlib imshow."""
        return self.tensor.permute(1, 2, 0).cpu().squeeze().numpy()

    @property
    def _cmap_imagen(self) -> str | None:
        return "gray" if self.is_grayscale else None

    @staticmethod
    def _validar_size(size: int) -> int:
        """Valida que el tamaño sea entero impar ≥1; ajusta al impar siguiente si es par."""
        if not isinstance(size, int):
            raise TypeError(f"size debe ser int, recibido {type(size).__name__}")
        if size < 1:
            raise ValueError(f"size debe ser ≥ 1, recibido {size}")
        if size % 2 == 0:
            ajustado = size + 1
            log.warning(f"size={size} es par; ajustando a {ajustado} (kernels requieren impar).")
            return ajustado
        return size

    @staticmethod
    def _validar_kernel_normalizado(kernel: np.ndarray, nombre: str = "kernel") -> None:
        """Avisa si el kernel no suma 1 (puede saturar el clamp [0,1])."""
        suma = kernel.sum()
        if abs(suma - 1.0) > TOLERANCIA_KERNEL_SUMA:
            log.warning(f"{nombre} suma {suma:.3f} (≠ 1); la salida puede salirse de [0,1].")

    @staticmethod
    def _kernel_box_1d(size: int) -> np.ndarray:
        """Kernel 1D de box filter (todos los pesos iguales, suma 1)."""
        return np.ones(size) / size

    @staticmethod
    def _kernel_gaussiano_1d(size: int, sigma: float | None = None) -> np.ndarray:
        """Kernel 1D gaussiano normalizado. Sigma auto vía fórmula OpenCV si es None."""
        if sigma is None:
            sigma = 0.3 * ((size - 1) / 2 - 1) + SIGMA_AUTO_BASE
        eje = np.arange(size) - size // 2
        kernel = np.exp(-(eje ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    @staticmethod
    def _kernel_piramidal_2d(size: int) -> np.ndarray:
        """Kernel 2D piramidal (Chebyshev): peso = center - dist + 1, normalizado."""
        center = size // 2
        kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist_chebyshev = max(abs(i - center), abs(j - center))
                kernel[i, j] = center - dist_chebyshev + 1
        return kernel / kernel.sum()

    def _construir_titulo(self, prefijo: str, titulo_usuario: str | None) -> str:
        """Compone el título del nodo derivado: '<usuario|prefijo> de <self.title>'."""
        cabeza = titulo_usuario if titulo_usuario is not None else prefijo
        return f"{cabeza} de {self.title}"

    def _aplicar_por_canal(self, fn_canal) -> torch.Tensor:
        """
        Aplica fn_canal a cada canal independientemente y concatena el resultado.
        fn_canal recibe un tensor (1, H, W) y debe devolver un tensor (1, H, W).
        Aplica clamp final a [0,1].
        """
        resultados = [fn_canal(self.tensor[c:c + 1]) for c in range(self.tensor.shape[0])]
        return torch.cat(resultados, dim=0).clamp(0.0, 1.0)

    @staticmethod
    def _fft_fila(canal_cpu: torch.Tensor, fila: int, excluir_dc: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        FFT 1D de una fila (señal real). Por simetría hermítica retorna solo
        la mitad positiva del espectro. Si excluir_dc, omite la frecuencia 0.
        """
        senal = canal_cpu[fila, :].numpy()
        magnitud = np.abs(np.fft.fft(senal))[: senal.size // 2]
        inicio = 1 if excluir_dc else 0
        return np.arange(inicio, magnitud.size), magnitud[inicio:]

    @staticmethod
    def _dft_magnitud_log(canal_np: np.ndarray) -> np.ndarray:
        """DFT 2D centrada en escala log normalizada a [0,1] para visualización."""
        espectro = np.fft.fftshift(np.fft.fft2(canal_np))
        magnitud = np.log1p(np.abs(espectro))
        pico = magnitud.max()
        return magnitud / pico if pico > 0 else magnitud

    # ──────────────────────────────────────────────────────────
    # Convolución
    # ──────────────────────────────────────────────────────────

    @tag(tipo="transformacion", hace="Convolución 2D por canal con padding reflect.",
         depende_de=("tensor", "F.conv2d", "F.pad", "_aplicar_por_canal"))
    def convolucion(self, kernel: np.ndarray, titulo: str | None = None,
                    flip_kernel: bool = False) -> Self:
        """
        Aplica un kernel sobre la imagen mediante convolución 2D con padding reflect.

        Nota: PyTorch implementa correlación cruzada (no convolución pura). Para kernels
        simétricos (gaussiano, box, piramidal) son equivalentes. Para asimétricos
        (ej. Sobel) usa flip_kernel=True para obtener convolución matemática.
        """
        if kernel.ndim != 2:
            raise ValueError(f"kernel debe ser 2D, recibido {kernel.ndim}D")
        alto_kernel, ancho_kernel = kernel.shape
        if alto_kernel % 2 == 0 or ancho_kernel % 2 == 0:
            raise ValueError(f"kernel debe tener dimensiones impares, recibido {alto_kernel}x{ancho_kernel}")
        self._validar_kernel_normalizado(kernel)

        if flip_kernel:
            kernel = kernel[::-1, ::-1].copy()

        pad_y, pad_x = alto_kernel // 2, ancho_kernel // 2
        peso = torch.tensor(kernel, dtype=torch.float32, device=self.tensor.device)
        peso = peso.unsqueeze(0).unsqueeze(0)

        def aplicar(canal: torch.Tensor) -> torch.Tensor:
            con_padding = F.pad(canal.unsqueeze(0), (pad_x, pad_x, pad_y, pad_y), mode="reflect")
            return F.conv2d(con_padding, peso).squeeze(0)

        resultado = self._aplicar_por_canal(aplicar)
        titulo_final = self._construir_titulo(f"Convolución ({alto_kernel}x{ancho_kernel})", titulo)
        return self.__class__(resultado, title=titulo_final)

    @tag(tipo="transformacion",
         hace="Convolución separable: kernel 1D horizontal + vertical (O(s) vs O(s²)).",
         depende_de=("tensor", "F.conv2d", "F.pad", "_aplicar_por_canal"))
    def convolucion_separable(self, kernel_1d: np.ndarray, titulo: str | None = None) -> Self:
        """
        Convolución separable 2D: aplica un kernel 1D primero en X y luego en Y.
        Equivalente a convolver con el producto externo k1d ⊗ k1d, pero O(N·s)
        en vez de O(N·s²). Solo válido para kernels separables (box, gaussiano).
        """
        if kernel_1d.ndim != 1:
            raise ValueError(f"kernel_1d debe ser 1D, recibido {kernel_1d.ndim}D")
        size = kernel_1d.size
        if size % 2 == 0:
            raise ValueError(f"kernel debe tener tamaño impar, recibido {size}")
        self._validar_kernel_normalizado(kernel_1d, nombre="kernel_1d")

        pad = size // 2
        device = self.tensor.device
        peso_horizontal = torch.tensor(kernel_1d, dtype=torch.float32, device=device).view(1, 1, 1, size)
        peso_vertical   = torch.tensor(kernel_1d, dtype=torch.float32, device=device).view(1, 1, size, 1)

        def aplicar(canal: torch.Tensor) -> torch.Tensor:
            tmp = canal.unsqueeze(0)
            tmp = F.pad(tmp, (pad, pad, 0, 0), mode="reflect")
            tmp = F.conv2d(tmp, peso_horizontal)
            tmp = F.pad(tmp, (0, 0, pad, pad), mode="reflect")
            tmp = F.conv2d(tmp, peso_vertical)
            return tmp.squeeze(0)

        resultado = self._aplicar_por_canal(aplicar)
        titulo_final = self._construir_titulo(f"Conv separable ({size})", titulo)
        return self.__class__(resultado, title=titulo_final)

    # ──────────────────────────────────────────────────────────
    # Filtros de alto nivel (fluent API)
    # ──────────────────────────────────────────────────────────

    @tag(tipo="transformacion", hace="Box filter separable.",
         depende_de=("convolucion_separable", "_kernel_box_1d"))
    def suavizar(self, size: int = 3) -> Self:
        """Suavizado por promedio uniforme (box filter), implementación separable."""
        size = self._validar_size(size)
        return self.convolucion_separable(self._kernel_box_1d(size), f"Suavizado {size}x{size}")

    @tag(tipo="transformacion", hace="Pasa-bajos piramidal (Chebyshev).",
         depende_de=("convolucion", "_kernel_piramidal_2d"))
    def piramidal(self, size: int = 3) -> Self:
        """Pasa-bajos piramidal: pesos decrecen linealmente desde el centro."""
        size = self._validar_size(size)
        return self.convolucion(self._kernel_piramidal_2d(size), f"Piramidal {size}x{size}")

    @tag(tipo="transformacion", hace="Gaussiano separable G(x)·G(y).",
         depende_de=("convolucion_separable", "_kernel_gaussiano_1d"))
    def gaussiano(self, size: int = 3, sigma: float | None = None) -> Self:
        """Filtro gaussiano separable. G(x,y)=e^(-(x²+y²)/2σ²) = G(x)·G(y)."""
        size = self._validar_size(size)
        return self.convolucion_separable(self._kernel_gaussiano_1d(size, sigma),
                                          f"Gaussiano {size}x{size}")

    # ──────────────────────────────────────────────────────────
    # Visualización: señal 1D + FFT por canal (con slider y lupa)
    # ──────────────────────────────────────────────────────────

    @tag(tipo="grafica", hace="Señal 1D + FFT por canal con slider y lupa.",
         depende_de=("tensor", "is_grayscale"))
    def senal_por_canal(self, fila: int | None = None, block: bool = False,
                        excluir_dc: bool = False) -> Self:
        """
        Por cada canal: señal de la fila seleccionada + su FFT.
        Slider para mover la fila; lupa que sigue al cursor sobre la imagen.
        """
        alto, ancho = self.height, self.width
        fila_inicial = alto // 2 if fila is None else fila

        canales = self._canales_para_lineas()
        n_canales = len(canales)
        imagen = self._imagen_para_imshow()
        tensor_cpu = self.tensor.cpu()

        fig, gs = self._crear_figura_senal(n_canales)
        ax_imagen, hline_imagen = self._dibujar_imagen_principal(fig, gs, imagen, fila_inicial)
        lupa = self._anadir_lupa(ax_imagen, imagen)
        plot_lineas = self._dibujar_filas_senal_fft(fig, gs, canales, tensor_cpu,
                                                     fila_inicial, ancho, excluir_dc)
        slider = self._anadir_slider_fila(fig, gs, alto, fila_inicial)

        def actualizar_por_fila(valor):
            fila_actual = int(valor)
            hline_imagen.set_ydata([fila_actual, fila_actual])
            ax_imagen.set_title(f"{self.title} (fila {fila_actual})", fontsize=11)
            self._actualizar_plots_senal(plot_lineas, tensor_cpu, fila_actual, excluir_dc)
            fig.canvas.draw_idle()

        slider.on_changed(actualizar_por_fila)
        cid_lupa = self._conectar_evento_lupa(fig, ax_imagen, imagen, lupa, alto, ancho)

        fig.suptitle(f"Señal y FFT por canal: {self.title}", fontsize=13, weight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        setattr(fig, "_slider_ref", slider)
        setattr(fig, "_lupa_cid", cid_lupa)
        plt.show(block=block)
        return self

    @staticmethod
    def _crear_figura_senal(n_canales: int) -> tuple[Figure, GridSpec]:
        n_filas = 1 + n_canales
        fig = plt.figure(figsize=(14, 3 * n_filas + 0.8))
        gs = GridSpec(n_filas + 1, 2, figure=fig,
                      height_ratios=[1.2] + [1] * n_canales + [0.3])
        return fig, gs

    def _dibujar_imagen_principal(self, fig, gs, imagen: np.ndarray, fila: int):
        ax = fig.add_subplot(gs[0, :])
        ax.imshow(imagen, cmap=self._cmap_imagen, vmin=0, vmax=1)
        hline = ax.axhline(y=fila, color=COLOR_FILA_REFERENCIA, linestyle="--", lw=1.5, alpha=0.9)
        ax.set_title(f"{self.title} (fila {fila})", fontsize=11)
        ax.axis("off")
        return ax, hline

    @staticmethod
    def _anadir_lupa(ax_imagen, imagen: np.ndarray) -> dict:
        """Crea inset 'lupa' en la esquina y devuelve refs para actualizar luego."""
        alto_img, ancho_img = imagen.shape[:2]
        radio = max(10, min(alto_img, ancho_img) // 20)

        ax = ax_imagen.inset_axes([0.82, 0.62, 0.18, 0.36])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Lupa", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_LUPA)
            spine.set_linewidth(1.5)

        cmap = "gray" if imagen.ndim == 2 else None
        im = ax.imshow(imagen[:2 * radio, :2 * radio], cmap=cmap, vmin=0, vmax=1)
        rect = Rectangle((0, 0), 2 * radio, 2 * radio,
                         fill=False, edgecolor=COLOR_LUPA, lw=1.2, alpha=0.8, visible=False)
        ax_imagen.add_patch(rect)
        return {"radio": radio, "imagen": im, "rect": rect}

    def _dibujar_filas_senal_fft(self, fig, gs, canales: list[CanalEspec],
                                  tensor_cpu: torch.Tensor, fila: int,
                                  ancho: int, excluir_dc: bool) -> list[dict]:
        """Dibuja una fila por canal: [señal | FFT]. Devuelve refs para actualizar."""
        eje_pixel = np.arange(ancho)
        plot_refs = []
        for i, canal in enumerate(canales):
            senal = tensor_cpu[canal.indice, fila, :].numpy() * INTENSIDAD_MAX
            freqs, magnitud = self._fft_fila(tensor_cpu[canal.indice], fila, excluir_dc)

            ax_senal = fig.add_subplot(gs[1 + i, 0])
            linea_senal, = ax_senal.plot(eje_pixel, senal, color=canal.color, lw=1)
            self._configurar_eje_senal(ax_senal, canal.nombre, ancho)

            ax_fft = fig.add_subplot(gs[1 + i, 1])
            linea_fft, = ax_fft.plot(freqs, magnitud, color=canal.color, lw=0.8)
            self._configurar_eje_fft(ax_fft, canal.nombre)

            plot_refs.append({"canal": canal, "linea_senal": linea_senal,
                              "linea_fft": linea_fft, "ax_fft": ax_fft})
        return plot_refs

    @staticmethod
    def _configurar_eje_senal(ax, nombre_canal: str, ancho: int) -> None:
        ax.set_title(f"Señal — Canal {nombre_canal}", fontsize=10)
        ax.set_xlabel("Pixel")
        ax.set_ylabel(f"Intensidad [0–{INTENSIDAD_MAX}]")
        ax.set_xlim(0, ancho - 1)
        ax.set_ylim(0, INTENSIDAD_MAX + 5)
        ax.grid(True, linestyle=":", alpha=0.4)

    @staticmethod
    def _configurar_eje_fft(ax, nombre_canal: str) -> None:
        ax.set_title(f"FFT — Canal {nombre_canal}", fontsize=10)
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel("Magnitud (log)")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.4)

    @staticmethod
    def _anadir_slider_fila(fig, gs, alto: int, fila_inicial: int) -> Slider:
        ax_slider = fig.add_subplot(gs[-1, :])
        return Slider(ax_slider, "Fila", 0, alto - 1, valinit=fila_inicial,
                      valstep=1, color="gold")

    def _actualizar_plots_senal(self, plot_refs: list[dict], tensor_cpu: torch.Tensor,
                                 fila: int, excluir_dc: bool) -> None:
        for ref in plot_refs:
            idx = ref["canal"].indice
            senal = tensor_cpu[idx, fila, :].numpy() * INTENSIDAD_MAX
            ref["linea_senal"].set_ydata(senal)
            _, magnitud = self._fft_fila(tensor_cpu[idx], fila, excluir_dc)
            ref["linea_fft"].set_ydata(magnitud)
            ref["ax_fft"].relim()
            ref["ax_fft"].autoscale_view()

    @staticmethod
    def _conectar_evento_lupa(fig, ax_imagen, imagen: np.ndarray, lupa: dict,
                               alto: int, ancho: int) -> int:
        radio = lupa["radio"]

        def on_move(event):
            if event.inaxes != ax_imagen or event.xdata is None or event.ydata is None:
                return
            cx, cy = int(event.xdata), int(event.ydata)
            x0 = max(0, min(ancho - 2 * radio, cx - radio))
            y0 = max(0, min(alto  - 2 * radio, cy - radio))
            recorte = imagen[y0:y0 + 2 * radio, x0:x0 + 2 * radio]
            lupa["imagen"].set_data(recorte)
            lupa["rect"].set_xy((x0, y0))
            lupa["rect"].set_visible(True)
            fig.canvas.draw_idle()

        return fig.canvas.mpl_connect("motion_notify_event", on_move)

    # ──────────────────────────────────────────────────────────
    # Visualización: comparación de FFT entre dos nodos
    # ──────────────────────────────────────────────────────────

    @tag(tipo="grafica", hace="Compara FFT por canal de dos nodos solapados con slider de fila.",
         depende_de=("tensor", "is_grayscale"))
    def comparar_fft(self, otro: "SignalVisionNode", fila: int | None = None,
                     block: bool = False, excluir_dc: bool = False) -> Self:
        """
        Solapa las magnitudes FFT (por canal) de dos imágenes.
        Línea sólida = self, línea punteada = otro. Slider compartido para mover la fila.
        """
        if self.tensor.shape != otro.tensor.shape:
            raise ValueError(f"Tensores incompatibles: {self.tensor.shape} vs {otro.tensor.shape}")

        alto = self.height
        fila_inicial = alto // 2 if fila is None else fila

        canales = self._canales_para_lineas()
        tensor_self = self.tensor.cpu()
        tensor_otro = otro.tensor.cpu()

        fig, gs = self._crear_figura_comparacion(len(canales))
        hlines = self._dibujar_par_imagenes(fig, gs, otro, fila_inicial)
        ejes_fft, lineas = self._dibujar_fft_solapadas(
            fig, gs, canales, tensor_self, tensor_otro, fila_inicial, excluir_dc, otro.title
        )
        slider = self._anadir_slider_fila(fig, gs, alto, fila_inicial)

        def actualizar(valor):
            fila_actual = int(valor)
            for hline in hlines:
                hline.set_ydata([fila_actual, fila_actual])
            for canal, par in zip(canales, lineas):
                _, mag_self = self._fft_fila(tensor_self[canal.indice], fila_actual, excluir_dc)
                _, mag_otro = self._fft_fila(tensor_otro[canal.indice], fila_actual, excluir_dc)
                par["linea_self"].set_ydata(mag_self)
                par["linea_otro"].set_ydata(mag_otro)
            for ax in ejes_fft:
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw_idle()

        slider.on_changed(actualizar)
        fig.suptitle(f"FFT comparativa: {self.title}  vs  {otro.title}", fontsize=12, weight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        setattr(fig, "_slider_ref", slider)
        plt.show(block=block)
        return self

    @staticmethod
    def _crear_figura_comparacion(n_canales: int) -> tuple[Figure, GridSpec]:
        fig = plt.figure(figsize=(5 * n_canales, 7))
        gs = GridSpec(3, n_canales, figure=fig, height_ratios=[1.2, 1.4, 0.15])
        return fig, gs

    def _dibujar_par_imagenes(self, fig, gs, otro: "SignalVisionNode", fila: int) -> list:
        """Dibuja self y otro lado a lado en la fila superior. Devuelve [hline_self, hline_otro]."""
        gs_top = gs[0, :].subgridspec(1, 2)
        ax_self = fig.add_subplot(gs_top[0, 0])
        ax_otro = fig.add_subplot(gs_top[0, 1])

        ax_self.imshow(self._imagen_para_imshow(), cmap=self._cmap_imagen, vmin=0, vmax=1)
        ax_otro.imshow(otro._imagen_para_imshow(), cmap=otro._cmap_imagen, vmin=0, vmax=1)

        hline_self = ax_self.axhline(y=fila, color=COLOR_FILA_REFERENCIA, linestyle="-",  lw=1.5, alpha=0.9)
        hline_otro = ax_otro.axhline(y=fila, color=COLOR_FILA_REFERENCIA, linestyle="--", lw=1.5, alpha=0.9)

        ax_self.set_title(f"{self.title} (—)",  fontsize=10)
        ax_self.axis("off")
        ax_otro.set_title(f"{otro.title} (--)", fontsize=10)
        ax_otro.axis("off")
        return [hline_self, hline_otro]

    def _dibujar_fft_solapadas(self, fig, gs, canales: list[CanalEspec],
                                tensor_self: torch.Tensor, tensor_otro: torch.Tensor,
                                fila: int, excluir_dc: bool,
                                etiqueta_otro: str) -> tuple[list, list[dict]]:
        ejes = [fig.add_subplot(gs[1, i]) for i in range(len(canales))]
        lineas = []
        for ax, canal in zip(ejes, canales):
            freqs, mag_self = self._fft_fila(tensor_self[canal.indice], fila, excluir_dc)
            _,     mag_otro = self._fft_fila(tensor_otro[canal.indice], fila, excluir_dc)

            linea_self, = ax.plot(freqs, mag_self, color=canal.color, lw=1.4, alpha=0.9, label=self.title)
            linea_otro, = ax.plot(freqs, mag_otro, color=canal.color, lw=1.0, linestyle="--",
                                  alpha=0.7, label=etiqueta_otro)
            self._configurar_eje_fft(ax, canal.nombre)
            ax.legend(fontsize=8, loc="upper right")
            lineas.append({"linea_self": linea_self, "linea_otro": linea_otro})
        return ejes, lineas

    # ──────────────────────────────────────────────────────────
    # Visualización: DFT 2D por canal
    # ──────────────────────────────────────────────────────────

    @tag(tipo="grafica", hace="DFT 2D por canal (fft2+fftshift+log).",
         depende_de=("tensor", "is_grayscale"))
    def transformada_fourier_2d(self, block: bool = False) -> Self:
        """
        DFT 2D por canal con layout tipo histograma().
        Color: 2x2 → imagen | DFT R / DFT G | DFT B
        Grayscale: 1x2 → imagen | DFT B/N
        """
        canales = self._canales_para_dft()
        ax_imagen, ejes_dft, fig = self._crear_layout_dft(len(canales))

        ax_imagen.imshow(self._imagen_para_imshow(), cmap=self._cmap_imagen, vmin=0, vmax=1)
        ax_imagen.set_title(self.title, fontsize=11)
        ax_imagen.axis("off")

        for ax, canal in zip(ejes_dft, canales):
            magnitud = self._dft_magnitud_log(self.tensor[canal.indice].cpu().numpy())
            ax.imshow(magnitud, cmap=canal.color, vmin=0, vmax=1)
            ax.set_title(f"DFT 2D — Canal {canal.nombre}", fontsize=11)
            ax.axis("off")

        fig.suptitle(f"Transformada de Fourier 2D: {self.title}", fontsize=13, weight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show(block=block)
        return self

    def _crear_layout_dft(self, _n_canales: int):
        """Devuelve (ax_imagen, [ax_dft...], fig) según si es grayscale o color."""
        if self.is_grayscale:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            return axes[0], [axes[1]], fig

        fig = plt.figure(figsize=(12, 10))
        ax_imagen = fig.add_subplot(2, 2, 1)
        ejes_dft = [fig.add_subplot(2, 2, i) for i in (2, 3, 4)]
        return ax_imagen, ejes_dft, fig


# ==========================================
# Demo / Test
# ==========================================
def demo_senales():
    try:
        # img_path = get_image_path("golf.bmp")
        img_path = get_image_path("texto.bmp")
        # img_path = get_image_path("patrones/senoidal.bmp")
        if not img_path.exists():
            log.warning("No se encontró la imagen de prueba.")
            return

        base = SignalVisionNode.desde_archivo(img_path)
        base.title = "Original"
        base.mostrar(block=False)

        log.step("1. Señal por canal (fila central)...")
        base.senal_por_canal()

        log.step("2. Transformada de Fourier 2D...")
        base.transformada_fourier_2d()

        log.step("3. Filtros: suavizado, gaussiano...")
        size = 11
        base_suavizado = base.suavizar(size=size)
        base_suavizado.mostrar(block=False)

        base_gaussiano = base.gaussiano(size=size)
        base_gaussiano.mostrar(block=False)

        base_gaussiano.transformada_fourier_2d(block=False)
        base_gaussiano.senal_por_canal(block=False)
        base_gaussiano.comparar_fft(base_suavizado, block=False)

        log.info("Demo completo. Cierra las ventanas para finalizar.")
        plt.show()

    except Exception as e:
        log.error(f"Error en el demo: {e}")


if __name__ == "__main__":
    demo_senales()
