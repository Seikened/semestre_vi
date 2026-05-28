"""
Web app de análisis frecuencial — proyecto del costal
=====================================================
Interfaz Streamlit para afinar parámetros de notch adaptativo en vivo,
muestra por muestra. Cada muestra tiene SUS propios params + opcionalmente
un pre-filtro espacial (gaussiano, mediana, gamma, etc.) que se aplica
antes del análisis FFT.

Layout:
    Sidebar: imagen · muestras · pre-filtro · sliders detección · métricas · botones
    Main:
      ┌─ ENTRADA ─────────┬─ SALIDA ──────────┐    ← grandes, lado a lado
      │ (con pre-filtro)  │ (sin tejido)      │
      └───────────────────┴───────────────────┘
      [FFT antes] [máscara] [FFT después] [diff ×5]   ← chicas, informativas

Persistencia:
    salida_dir/params_por_muestra.json:
    {
      "1": {
        "deteccion": {umbral_relativo, min_radio, distancia_min, banda_v},
        "pre_filtro": {"nombre": "gaussiano", "args": {size: 5, sigma: 1.0}}
      }
    }

Cómo correr:
    uv run streamlit run image_processing/proyecto_final/web_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_aqui = Path(__file__).resolve().parent
_project_root = _aqui.parents[1]
for _p in (str(_project_root), str(_aqui)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_image_comparison import image_comparison  # noqa: E402

from image_processing import DerivativeVisionNode  # noqa: E402

from analisis_muestras import (  # noqa: E402
    PARAMS_PICOS_DEFAULT,
    analizar_muestra,
    combinar_picos_a_imagen,
)
from notch import (  # noqa: E402
    aplicar_notch_adaptativo,
    detectar_picos_ambos_ejes,
)
from selector_muestras import (  # noqa: E402
    Muestra,
    cargar_muestras,
    cargar_params_por_muestra,
    guardar_params_por_muestra,
)
from web_lib import (  # noqa: E402
    DETECCION_KEYS,
    FILTROS,
    aplicar_notch_con_mascara_diff,
    aplicar_notch_con_preservacion,
    aplicar_notch_con_preservacion_combinada,
    aplicar_pre_filtro,
    calcular_pipeline,
    componer_tensor_con_salida,
    desempacar_params,
    empacar_params,
    fig_espectro_picos,
    fig_espectro_simple,
    fig_imagen,
    fig_imagen_completa_con_bboxes,
    tensor_a_arr,
    tensor_a_uint8_hwc,
)


st.set_page_config(
    page_title="Análisis Costal",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────
# Cache de IO (decoradores Streamlit, no se pueden mover a web_lib.py)
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando imagen…")
def cargar_imagen(imagen_path_str: str):
    return DerivativeVisionNode.desde_archivo(Path(imagen_path_str))


@st.cache_resource(show_spinner="Cargando muestras…")
def cargar_muestras_cached(salida_dir_str: str, _firma: str):
    return cargar_muestras(Path(salida_dir_str))


# ──────────────────────────────────────────────────────────────────
# Widgets de sidebar
# ──────────────────────────────────────────────────────────────────

def slider_con_num(label, mn, mx, default, step, key,
                    *, fmt=None, help=None, contenedor=None):
    """
    Slider + number_input sincronizados (vía session_state + on_change).
    Permite arrastrar O escribir el valor exacto. Devuelve el valor actual.

    `contenedor` puede ser st.sidebar para colocar los widgets allí.
    """
    box = contenedor if contenedor is not None else st

    sl_key = f"{key}__sl"
    num_key = f"{key}__num"

    if sl_key not in st.session_state:
        st.session_state[sl_key] = default
    if num_key not in st.session_state:
        st.session_state[num_key] = default

    def _from_sl():
        st.session_state[num_key] = st.session_state[sl_key]

    def _from_num():
        st.session_state[sl_key] = st.session_state[num_key]

    c1, c2 = box.columns([3, 1], gap="small")
    with c1:
        st.slider(label, min_value=mn, max_value=mx, step=step,
                    key=sl_key, on_change=_from_sl, help=help)
    with c2:
        st.number_input(label, min_value=mn, max_value=mx, step=step,
                          key=num_key, on_change=_from_num,
                          format=fmt, label_visibility="hidden")

    return st.session_state[sl_key]


def widget_pre_filtro(filtro_inicial: str = "ninguno",
                       args_iniciales: Optional[dict] = None) -> tuple[str, dict]:
    """Selector de filtro + sliders+input de sus args."""
    nombres = list(FILTROS.keys())
    idx = nombres.index(filtro_inicial) if filtro_inicial in nombres else 0
    nombre = st.sidebar.selectbox(
        "Filtro previo a la muestra",
        nombres, index=idx,
        help="Se aplica a la muestra ANTES del análisis FFT. "
             "Afecta entrada y salida.",
    )
    args: dict = {}
    args_iniciales = args_iniciales or {}
    for spec in FILTROS[nombre]["args"]:
        nombre_arg, tipo, mn, mx, default, step = spec
        valor_inicial = args_iniciales.get(nombre_arg, default)
        if tipo == "int":
            args[nombre_arg] = slider_con_num(
                f"  · {nombre_arg}",
                int(mn), int(mx), int(valor_inicial), int(step),
                key=f"flt_{nombre}_{nombre_arg}",
                contenedor=st.sidebar,
            )
        else:  # float
            args[nombre_arg] = slider_con_num(
                f"  · {nombre_arg}",
                float(mn), float(mx), float(valor_inicial), float(step),
                key=f"flt_{nombre}_{nombre_arg}",
                fmt="%.2f",
                contenedor=st.sidebar,
            )
    return nombre, args


def _aplicar_preset(presets: dict[str, float | int]) -> None:
    """Setea los keys del session_state para los sliders sincronizados (slider_con_num)."""
    for key, val in presets.items():
        st.session_state[f"{key}__sl"] = val
        st.session_state[f"{key}__num"] = val


def widget_sliders_deteccion(params_iniciales: dict) -> dict:
    """Sliders+input de detección de picos en el sidebar (rangos amplios)."""

    # Botones de configuración rápida — un click setea los 3 sliders
    st.sidebar.markdown("**🎯 Configs probadas**")
    cb1, cb2, cb3 = st.sidebar.columns(3, gap="small")
    if cb1.button("🛡 Suave", use_container_width=True,
                    help="Pocos picos, solo los obvios. Tejido se elimina menos pero "
                          "no toca otras freqs."):
        _aplicar_preset({"det_umbral": 0.15, "det_min_radio": 30, "det_dist_min": 6})
        st.rerun()
    if cb2.button("⚖️ Medio", use_container_width=True,
                    help="Balance — default. Buen punto de partida."):
        _aplicar_preset({"det_umbral": 0.10, "det_min_radio": 20, "det_dist_min": 4})
        st.rerun()
    if cb3.button("⚡ Fuerte", use_container_width=True,
                    help="Atrapa más picos = más tejido eliminado, también más artefactos."):
        _aplicar_preset({"det_umbral": 0.05, "det_min_radio": 10, "det_dist_min": 3})
        st.rerun()

    with st.sidebar.expander("🤔 ¿Qué muevo para qué?"):
        st.markdown(
            "**Objetivo**: identificar qué frecuencias del FFT son del tejido "
            "(para luego cancelarlas).\n\n"
            "**Si la SALIDA aún tiene tejido visible:**\n"
            "- ⬇ baja `Umbral` (atrapa picos más débiles)\n"
            "- ⬇ baja `Radio mín.` (atrapa picos cerca del centro del espectro)\n\n"
            "**Si detectas demasiados picos (50+ en muestra de 256×256):**\n"
            "- ⬆ sube `Umbral` (más estricto)\n"
            "- ⬆ sube `Radio mín.` (ignora ruido cercano al DC = iluminación)\n\n"
            "**Si los picos parecen apelotonados:**\n"
            "- ⬇ baja `Distancia mín.`\n\n"
            "**Truco visual:** mira el panel `FFT antes` — los círculos rojos "
            "deben caer sobre los puntos brillantes del espectro. Si caen en "
            "zonas oscuras, los params están mal."
        )
    umbral = slider_con_num(
        "Umbral % del máximo",
        0.0, 1.0,
        float(params_iniciales.get("umbral_relativo", 0.10)),
        0.01,
        key="det_umbral", fmt="%.2f", contenedor=st.sidebar,
        help="Solo dispara si el pico supera este porcentaje del pico más alto del eje.",
    )
    min_radio = slider_con_num(
        "Radio mín. (excluir DC)",
        0, 500,
        int(params_iniciales.get("min_radio", 20)),
        1,
        key="det_min_radio", contenedor=st.sidebar,
        help="Excluye disco alrededor del DC. Sube para descartar iluminación/fondo.",
    )
    dist_min = slider_con_num(
        "Distancia mín. entre picos",
        1, 200,
        int(params_iniciales.get("distancia_min", 4)),
        1,
        key="det_dist_min", contenedor=st.sidebar,
    )
    return {
        "umbral_relativo": float(umbral),
        "min_radio": int(min_radio),
        "distancia_min": int(dist_min),
        "banda_v": 2,
    }


# ──────────────────────────────────────────────────────────────────
# Vistas principales
# ──────────────────────────────────────────────────────────────────

def vista_inicio(nodo, muestras, params_por_muestra=None):
    params_por_muestra = params_por_muestra or {}
    n_con_params = sum(1 for m in muestras if m.id in params_por_muestra)

    st.info("👈 Selecciona una muestra del sidebar para empezar.")

    tensor_override = None
    if n_con_params > 0:
        col_a, col_b = st.columns([2, 5])
        with col_a:
            ver_salida = st.toggle(
                "Ver parches con salida limpia",
                value=False,
                help=(
                    f"{n_con_params} muestra(s) tienen params guardados. "
                    "Activa para reemplazar cada parche por la salida del "
                    "pipeline (pre-filtro + notch + IFFT)."
                ),
            )
        with col_b:
            estado = (
                f"🟢 mostrando salida limpia compuesta sobre {n_con_params} parche(s)"
                if ver_salida else
                f"⚪ original ({n_con_params}/{len(muestras)} con params guardados)"
            )
            st.caption(estado)
        if ver_salida:
            with st.spinner("Calculando salidas por parche…"):
                tensor_override = componer_tensor_con_salida(
                    nodo, muestras, params_por_muestra
                )
    else:
        st.caption(
            "💡 Cuando guardes params para alguna muestra, aquí aparecerá un "
            "toggle para ver la salida limpia compuesta sobre la imagen completa."
        )

    # Título de la figura siempre estático — no desplaza la imagen al togglear
    st.pyplot(fig_imagen_completa_con_bboxes(
        nodo, muestras, None,
        tensor_override=tensor_override,
    ))


def vista_muestra_activa(muestra: Muestra, filtro_nombre: str,
                          filtro_args: dict, params_deteccion: dict):
    """Layout: 2 imágenes grandes lado a lado + 4 chicas abajo."""
    res = calcular_pipeline(muestra, filtro_nombre, filtro_args, params_deteccion)

    pre_label = ""
    if filtro_nombre != "ninguno":
        argstr = ", ".join(f"{k}={v}" for k, v in filtro_args.items())
        pre_label = f"  ·  pre-filtro: {filtro_nombre}({argstr})"

    st.subheader(
        f"#{muestra.id}  '{muestra.etiqueta}'  ·  "
        f"{muestra.tamano}×{muestra.tamano}{pre_label}"
    )

    # ── Top: ENTRADA vs SALIDA grandes lado a lado ──
    col_in, col_out = st.columns(2, gap="small")
    with col_in:
        arr_in, cmap_in = tensor_a_arr(res["muestra_pre"].tensor)
        sufijo_in = " (con pre-filtro)" if filtro_nombre != "ninguno" else ""
        st.pyplot(fig_imagen(
            arr_in, cmap_in,
            f"📥 ENTRADA{sufijo_in}",
            figsize=(7.5, 7.5),
        ))
    with col_out:
        st.pyplot(fig_imagen(
            res["img_filt"], "gray",
            "📤 SALIDA — sin tejido (IFFT)",
            figsize=(7.5, 7.5),
            color_titulo="darkgreen",
        ))

    # ── Bottom: 4 paneles informativos chicos ──
    st.markdown("**Información del análisis** (no editable)")
    F_log_p99 = max(float(np.percentile(np.log1p(res["F_mag"]), 99.5)), 1e-9)
    diff = np.abs(res["analisis"].luminance - res["img_filt"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.pyplot(fig_espectro_picos(
            res["F_mag"], res["picos"],
            f"FFT antes\n{len(res['picos'])} picos (rojo)",
            figsize=(3.5, 3.5),
        ))
    with c2:
        st.pyplot(fig_imagen(
            res["mascara"], "gray",
            "Máscara H(u,v)",
            figsize=(3.5, 3.5),
        ))
    with c3:
        st.pyplot(fig_espectro_simple(
            res["F_filt_mag"], F_log_p99,
            f"FFT después\n{res['pct']:.2f}% removido",
            figsize=(3.5, 3.5),
        ))
    with c4:
        st.pyplot(fig_imagen(
            np.clip(diff * 5.0, 0, 1), "hot",
            "Diferencia × 5",
            figsize=(3.5, 3.5),
        ))

    # ── Tabla de picos ──
    with st.expander(f"📊 Tabla de los {len(res['picos'])} picos"):
        if res["picos"]:
            N = muestra.tamano
            cy, cx = N // 2, N // 2
            rows = []
            for i, (u, v, mag, sigma) in enumerate(res["picos"], start=1):
                du, dv = u - cx, v - cy
                r = float(np.hypot(du, dv))
                T = N / r if r > 0 else float("inf")
                rows.append({
                    "#": i, "du": du, "dv": dv,
                    "T (px)": f"{T:.2f}" if T != float("inf") else "∞",
                    "|F|": f"{mag:.0f}",
                    "σ": f"{sigma:.2f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Sin picos detectados.")

    return res


def vista_aplicar_total(nodo, muestras, params_por_muestra):
    st.header("🚀 Resultado en imagen completa")
    if not params_por_muestra:
        st.error("No hay params guardados.")
        return

    with st.spinner("Re-detectando picos por muestra y combinando…"):
        analisis_list = []
        for m in muestras:
            if m.id not in params_por_muestra:
                continue
            det, pre_n, pre_args = desempacar_params(params_por_muestra[m.id])
            tensor_pre = aplicar_pre_filtro(m.tensor, pre_n, pre_args)
            m_pre = Muestra(
                id=m.id, centro_xy=m.centro_xy, tamano=m.tamano,
                etiqueta=m.etiqueta, fecha=m.fecha, tensor=tensor_pre,
            )
            an = analizar_muestra(m_pre)
            params_solo_det = {k: v for k, v in det.items() if k in DETECCION_KEYS}
            picos = detectar_picos_ambos_ejes(an.luminance, **params_solo_det)
            an.picos_2d = picos
            analisis_list.append(an)

        H, W = nodo.tensor.shape[1], nodo.tensor.shape[2]
        min_votos = 2 if len(analisis_list) >= 2 else 1
        picos_imagen = combinar_picos_a_imagen(
            analisis_list, shape_imagen=(H, W),
            min_votos=min_votos, epsilon=0.005,
        )

    st.metric(
        "Picos finales para la imagen completa", len(picos_imagen),
        help=f"Combinados de {len(analisis_list)} muestras (votos ≥ {min_votos}).",
    )

    if not picos_imagen:
        st.error("Ningún pico sobrevivió. Sube min_votos o afina más muestras.")
        return

    # ── Toggle: preservación de texto/bordes ──
    st.divider()
    col_toggle, col_help = st.columns([1, 3])
    with col_toggle:
        preservar = st.toggle(
            "🛡 Preservar texto/bordes",
            value=True,
            help=(
                "Detecta texto/bordes con binarización adaptativa y compone "
                "M·original + (1-M)·filtrada. Las letras se quedan intactas, "
                "el tejido se elimina solo en el fondo."
            ),
        )
    with col_help:
        if preservar:
            st.caption(
                "🛡 ON — texto/bordes intactos, tejido removido solo en fondo."
            )
        else:
            st.caption(
                "⚠️ OFF — notch aplicado a TODA la imagen (las letras pueden distorsionarse)."
            )

    if preservar:
        # ── Selector del TIPO de máscara ──
        st.markdown("**Tipo de máscara para preservar texto**")
        tipo_mask = st.radio(
            "tipo_mask",
            options=[
                "🧠 Diff inteligente (NUEVO — usa lo que el notch no afectó)",
                "🔤 Binarización adaptativa (detecta zonas de alto contraste)",
                "✨ Combinada (max de las dos)",
            ],
            index=0,
            label_visibility="collapsed",
            help=(
                "🧠 Diff: aprovecha que el notch elimina mucho en fondo y poco en "
                "impresiones. Captura letras + logo + escudo + barras + espigas SIN binarizar.\n\n"
                "🔤 Binarización: detección clásica por contraste local (la que ya conocías).\n\n"
                "✨ Combinada: usa la máscara más generosa (donde cualquiera detecte texto, preserva)."
            ),
        )

        usa_diff = "Diff" in tipo_mask
        usa_bin = "Binarización" in tipo_mask
        usa_combinada = "Combinada" in tipo_mask

        # ── Parámetros según tipo ──
        if usa_diff or usa_combinada:
            with st.expander("🧠 Parámetros máscara DIFF"):
                modo_diff = st.radio(
                    "Modo",
                    options=["lineal", "threshold"],
                    index=0, horizontal=True,
                    help=(
                        "lineal: máscara continua = 1 - diff_normalizada (suave).\n"
                        "threshold: binaria = 1 si diff < umbral, 0 si no (más cortante)."
                    ),
                )
                umbral_diff = slider_con_num(
                    "Umbral (solo modo threshold)",
                    0.0, 0.5, 0.05, 0.01,
                    key="diff_umbral", fmt="%.2f",
                    help="Pixeles con diff_normalizada menor a esto cuentan como impresión.",
                )
                suavizado_diff = slider_con_num(
                    "Suavizado σ (diff)",
                    0.0, 8.0, 2.0, 0.1,
                    key="diff_suavizado", fmt="%.1f",
                )

        if usa_bin or usa_combinada:
            st.markdown("**🎯 Configs rápidas (binarización)**")
            bp1, bp2, bp3 = st.columns(3, gap="small")
            if bp1.button("🛡 Proteger más", use_container_width=True):
                _aplicar_preset({
                    "mask_kernel": 51, "mask_c": 0.05,
                    "mask_dilatacion": 10, "mask_suavizado": 2.0,
                })
                st.rerun()
            if bp2.button("⚖️ Default", use_container_width=True):
                _aplicar_preset({
                    "mask_kernel": 51, "mask_c": 0.08,
                    "mask_dilatacion": 5, "mask_suavizado": 1.5,
                })
                st.rerun()
            if bp3.button("⚡ Filtrar más cerca", use_container_width=True):
                _aplicar_preset({
                    "mask_kernel": 31, "mask_c": 0.15,
                    "mask_dilatacion": 2, "mask_suavizado": 0.5,
                })
                st.rerun()

            with st.expander("🔤 Parámetros máscara BINARIZACIÓN"):
                mask_kernel = slider_con_num(
                    "Kernel binarización (px)", 11, 201, 51, 2,
                    key="mask_kernel",
                )
                mask_c = slider_con_num(
                    "Sensibilidad C", 0.01, 0.30, 0.08, 0.01,
                    key="mask_c", fmt="%.2f",
                )
                mask_dilatacion = slider_con_num(
                    "Dilatación (iteraciones)", 0, 20, 5, 1,
                    key="mask_dilatacion",
                )
                mask_suavizado = slider_con_num(
                    "Suavizado σ (binarización)", 0.0, 5.0, 1.5, 0.1,
                    key="mask_suavizado", fmt="%.1f",
                )

        # ── Aplicar según tipo elegido ──
        with st.spinner(f"Aplicando notch + máscara {tipo_mask.split()[1]}…"):
            if usa_diff:
                nodo_filt_raw, nodo_compuesto, mask_2d = aplicar_notch_con_mascara_diff(
                    nodo, picos_imagen,
                    modo=modo_diff,
                    umbral=umbral_diff,
                    suavizado_sigma=suavizado_diff,
                )
            elif usa_bin:
                nodo_filt_raw, nodo_compuesto, mask_2d = aplicar_notch_con_preservacion(
                    nodo, picos_imagen,
                    kernel_size=mask_kernel, c=mask_c,
                    dilatacion=mask_dilatacion,
                    suavizado_sigma=mask_suavizado,
                )
            else:  # combinada
                (nodo_filt_raw, nodo_compuesto, mask_2d,
                 _, _) = aplicar_notch_con_preservacion_combinada(
                    nodo, picos_imagen,
                    kernel_size=mask_kernel, c=mask_c,
                    dilatacion=mask_dilatacion,
                    suavizado_sigma_bin=mask_suavizado,
                    modo_diff=modo_diff, umbral_diff=umbral_diff,
                    suavizado_sigma_diff=suavizado_diff,
                )
        nodo_resultado = nodo_compuesto
    else:
        with st.spinner("Aplicando notch a la imagen completa…"):
            nodo_resultado = aplicar_notch_adaptativo(nodo, picos_imagen)
            nodo_filt_raw = nodo_resultado
            mask_2d = None

    # ── Comparador slider Antes/Después (deslizable) ──
    st.subheader("🔍 Comparador deslizable")
    st.caption(
        "Arrastra la línea vertical: izquierda = original, derecha = filtrado. "
        "Útil para ver con detalle qué cambió y qué se preservó."
    )
    img_antes_u8 = tensor_a_uint8_hwc(nodo.tensor)
    img_despues_u8 = tensor_a_uint8_hwc(nodo_resultado.tensor)
    sufijo = " · preservación ON" if preservar else " · preservación OFF"
    image_comparison(
        img1=img_antes_u8,  # type: ignore[arg-type]  # acepta numpy en runtime
        img2=img_despues_u8,  # type: ignore[arg-type]
        label1="📥 Antes (original)",
        label2=f"📤 Después{sufijo}",
        width=900,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )

    # ── Lado a lado tradicional (en expander, por si quieres ver completas) ──
    with st.expander("Ver antes/después lado a lado (sin slider)"):
        col1, col2 = st.columns(2)
        with col1:
            arr_orig, cmap_orig = tensor_a_arr(nodo.tensor)
            st.pyplot(fig_imagen(arr_orig, cmap_orig,
                                    "📥 Antes (original)", figsize=(7.5, 7.5)))
        with col2:
            arr_filt, cmap_filt = tensor_a_arr(nodo_resultado.tensor)
            st.pyplot(fig_imagen(arr_filt, cmap_filt,
                                    f"📤 Después{sufijo}",
                                    figsize=(7.5, 7.5),
                                    color_titulo="darkgreen"))

    # ── Lo que se eliminó + máscara ──
    diff = (nodo.tensor - nodo_resultado.tensor).abs().mean(dim=0).cpu().numpy()
    if preservar and mask_2d is not None:
        col_diff, col_mask = st.columns(2)
        with col_diff:
            st.pyplot(fig_imagen(np.clip(diff * 5.0, 0, 1), "hot",
                                    "Lo que se eliminó (× 5)",
                                    figsize=(7.5, 5.5)))
        with col_mask:
            st.pyplot(fig_imagen(mask_2d, "gray",
                                    "Máscara de texto (1=preservar, 0=filtrar)",
                                    figsize=(7.5, 5.5)))
        # Pequeño hint comparativo
        diff_raw = (nodo.tensor - nodo_filt_raw.tensor).abs().mean(dim=0).cpu().numpy()
        impacto_total = float(diff_raw.mean())
        impacto_real = float(diff.mean())
        st.caption(
            f"Impacto sin preservación: {impacto_total:.4f}  ·  "
            f"Impacto con preservación: {impacto_real:.4f}  "
            f"({100*(impacto_total - impacto_real)/max(impacto_total, 1e-9):.1f}% del cambio "
            "fue revertido en zonas de texto)"
        )
    else:
        st.pyplot(fig_imagen(np.clip(diff * 5.0, 0, 1), "hot",
                                "Lo que se eliminó (× 5)",
                                figsize=(11, 6)))


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

st.title("🧵 Análisis frecuencial del costal")
st.caption(
    "Cancelación del patrón de tejido vía FFT + notch adaptativo  ·  "
    "params propios por muestra  ·  pre-filtro espacial opcional"
)

# ── Sidebar: imagen + lista de muestras ──
with st.sidebar:
    st.header("📁 Imagen")
    imagenes = sorted(_aqui.glob("*.bmp"))
    if not imagenes:
        st.error(f"No hay BMPs en {_aqui}")
        st.stop()
    imagen_path = st.selectbox(
        "Imagen", imagenes, format_func=lambda p: p.name
    )

nodo = cargar_imagen(str(imagen_path))
salida_dir = _aqui / "muestras" / imagen_path.stem
muestras = cargar_muestras_cached(str(salida_dir), str(imagen_path))

if not muestras:
    st.warning(
        f"No hay muestras guardadas en `{salida_dir.relative_to(_project_root)}`."
    )
    st.markdown("Corre primero el selector:")
    st.code(
        "uv run python image_processing/proyecto_final/selector_muestras.py",
        language="bash",
    )
    st.stop()

params_por_muestra = cargar_params_por_muestra(salida_dir)

with st.sidebar:
    st.divider()
    st.header(f"🎯 Muestras ({len(muestras)})")
    st.caption("✓ guardado · ○ pendiente")

    activa_id = st.session_state.get("muestra_activa_id")
    aplicando = st.session_state.get("aplicar_total", False)

    for m in muestras:
        emoji = "✓" if m.id in params_por_muestra else "○"
        es_activa = (activa_id == m.id) and not aplicando
        if st.button(
            f"{emoji}  #{m.id}  '{m.etiqueta}'  ({m.tamano}px)",
            key=f"btn_m_{m.id}",
            use_container_width=True,
            type="primary" if es_activa else "secondary",
        ):
            st.session_state.muestra_activa_id = m.id
            st.session_state.pop("aplicar_total", None)
            st.rerun()

    n_listas = len(params_por_muestra)
    st.divider()
    if st.button(
        f"🚀 Aplicar a imagen completa  ({n_listas}/{len(muestras)} listas)",
        use_container_width=True, type="primary",
        disabled=n_listas == 0,
    ):
        st.session_state.aplicar_total = True
        st.session_state.pop("muestra_activa_id", None)
        st.rerun()

    st.caption(f"Tensor: {tuple(nodo.tensor.shape)}")
    st.caption(f"`{salida_dir.relative_to(_project_root)}`")


# ── Ramas de vista ──
if st.session_state.get("aplicar_total"):
    vista_aplicar_total(nodo, muestras, params_por_muestra)
    if st.button("← Volver al análisis por muestra"):
        st.session_state.pop("aplicar_total", None)
        st.rerun()
    st.stop()

if "muestra_activa_id" not in st.session_state:
    vista_inicio(nodo, muestras, params_por_muestra)
    st.stop()


# ── Hay muestra activa ──
muestra_id = st.session_state.muestra_activa_id
muestra = next(m for m in muestras if m.id == muestra_id)

# Cargar params guardados para esta muestra
params_guardados = params_por_muestra.get(muestra.id, {})
det_inicial, filtro_inicial, args_iniciales = desempacar_params(params_guardados)
if not det_inicial:
    det_inicial = dict(PARAMS_PICOS_DEFAULT)

# Sidebar específico de la muestra activa
with st.sidebar:
    st.divider()
    st.header("🔧 Pre-filtro")
    st.caption("se aplica a la muestra antes del análisis FFT")
    filtro_nombre, filtro_args = widget_pre_filtro(filtro_inicial, args_iniciales)

    st.divider()
    st.header("🎚 Detección de picos")
    params_deteccion = widget_sliders_deteccion(det_inicial)

# Renderizar vista (calcula y dibuja todo). Si algo truena, mostrar UI de error
# en vez de pantalla en blanco — copiable.
res_calc = None
try:
    res_calc = vista_muestra_activa(muestra, filtro_nombre, filtro_args, params_deteccion)
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    st.error(
        f"⚠️  **Error al procesar la muestra**\n\n"
        f"`{type(e).__name__}: {e}`\n\n"
        f"Esto puede ser por una combinación de pre-filtro / sliders no soportada. "
        f"Cambia algún slider o el filtro y reintenta."
    )
    with st.expander("📋 Traceback completo (selecciona y copia)", expanded=False):
        st.code(tb, language="python")

# Sidebar: métricas + botones (después de calcular)
with st.sidebar:
    st.divider()
    st.header("📈 Resultado")
    if res_calc is not None:
        st.metric("Picos detectados", len(res_calc["picos"]))
        st.metric("Energía removida", f"{res_calc['pct']:.2f}%")
    else:
        st.warning("Sin métricas: el cálculo falló.")

    payload_actual = empacar_params(params_deteccion, filtro_nombre, filtro_args)
    es_actual = (params_por_muestra.get(muestra.id) == payload_actual)
    estado = "✓ guardado" if es_actual else "○ sin guardar"
    st.caption(f"**Estado:** {estado}")

    if st.button("💾 Guardar params para esta muestra",
                  type="primary", use_container_width=True,
                  disabled=(es_actual or res_calc is None)):
        params_por_muestra[muestra.id] = payload_actual
        guardar_params_por_muestra(salida_dir, params_por_muestra)
        st.success(f"Guardado para #{muestra.id}")
        st.rerun()

    if st.button("← Volver al inicio (sin guardar)", use_container_width=True):
        st.session_state.pop("muestra_activa_id", None)
        st.rerun()
