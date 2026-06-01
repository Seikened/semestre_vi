"""Mildiu Scan — servidor FastAPI sobre el pipeline de procesar_hoja.

Arrancar:
    uv run uvicorn image_processing.canada.app:app --reload --port 8000

Sirve el frontend estático en `/`, expone SSE en `/api/stream` y procesa las 47
muestras en background. El estado vive en memoria; un cliente que se conecte
tarde recibe primero un snapshot con lo que ya está procesado.
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from colorstreak import Logger as log
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_processing.canada.main import DATA, OUT, fotos, procesar_hoja  # noqa: E402

WEB = Path(__file__).parent / "web"
OUT_BATCH = OUT / "batch"
CSV_PATH = OUT / "resumen.csv"

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    for q in list(estado.suscriptores):
        with suppress(Exception):
            q.put_nowait({"tipo": "shutdown"})
    estado.suscriptores.clear()


app = FastAPI(title="Mildiu Scan", lifespan=lifespan)


class Estado:
    """Estado compartido del batch: muestras, suscriptores SSE y flag de actividad."""

    def __init__(self) -> None:
        self.muestras: dict[str, dict] = {}
        self.suscriptores: set[asyncio.Queue] = set()
        self.procesando_activo: bool = False
        self.terminado: bool = False
        self.resumen: dict | None = None
        self.lock = asyncio.Lock()


estado = Estado()


def _muestras_iniciales() -> list[Path]:
    return [p for p in fotos() if p.exists()]


def _guardar_salidas(nombre: str, resultado: dict) -> None:
    OUT_BATCH.mkdir(parents=True, exist_ok=True)
    plt.imsave(OUT_BATCH / f"{nombre}_overlay.png", np.clip(resultado["overlay"], 0, 1))
    plt.imsave(OUT_BATCH / f"{nombre}_mascara.png", resultado["mascara"], cmap="gray", vmin=0, vmax=255)
    hongo_img = (resultado["hongo"].astype(np.uint8)) * 255
    plt.imsave(OUT_BATCH / f"{nombre}_hongo.png", hongo_img, cmap="gray", vmin=0, vmax=255)


def _escribir_csv() -> None:
    filas = [
        {
            "archivo": f"{nombre}.png",
            "pct_infeccion": round(m["pct"], 2),
            "area_hoja_px": m["area_hoja"],
            "area_hongo_px": m["area_hongo"],
        }
        for nombre, m in estado.muestras.items()
        if m.get("estado") == "listo"
    ]
    if not filas:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(filas).write_csv(CSV_PATH)


async def _publicar(msg: dict) -> None:
    for q in list(estado.suscriptores):
        await q.put(msg)


def _snapshot() -> dict:
    return {
        "tipo": "snapshot",
        "muestras": estado.muestras,
        "procesando_activo": estado.procesando_activo,
        "terminado": estado.terminado,
        "resumen": estado.resumen,
    }


async def _correr_batch() -> None:
    rutas = _muestras_iniciales()
    if not rutas:
        log.warning("No hay muestras en data/")
        estado.procesando_activo = False
        return

    log.step(f"Batch sobre {len(rutas)} hojas")
    loop = asyncio.get_running_loop()

    for ruta in rutas:
        nombre = ruta.stem
        estado.muestras[nombre] = {"estado": "procesando"}
        await _publicar({"tipo": "estado", "nombre": nombre, "estado": "procesando"})

        try:
            resultado = await loop.run_in_executor(None, procesar_hoja, ruta)
            await loop.run_in_executor(None, _guardar_salidas, nombre, resultado)
        except Exception as exc:
            log.error(f"{nombre}: {exc}")
            estado.muestras[nombre] = {"estado": "error", "mensaje": str(exc)}
            await _publicar({"tipo": "estado", "nombre": nombre, "estado": "error", "mensaje": str(exc)})
            continue

        estado.muestras[nombre] = {
            "estado": "listo",
            "pct": round(float(resultado["pct"]), 2),
            "area_hoja": int(resultado["area_hoja"]),
            "area_hongo": int(resultado["area_hongo"]),
        }
        await _publicar({
            "tipo": "estado",
            "nombre": nombre,
            "estado": "listo",
            "pct": estado.muestras[nombre]["pct"],
            "area_hoja": estado.muestras[nombre]["area_hoja"],
            "area_hongo": estado.muestras[nombre]["area_hongo"],
        })
        log.info(f"{nombre} → {estado.muestras[nombre]['pct']:.2f}%")

    _escribir_csv()
    listos = [m["pct"] for m in estado.muestras.values() if m.get("estado") == "listo"]
    if listos:
        estado.resumen = {
            "promedio": round(sum(listos) / len(listos), 2),
            "min": round(min(listos), 2),
            "max": round(max(listos), 2),
            "n": len(listos),
            "csv": "/out/resumen.csv",
        }
    estado.terminado = True
    estado.procesando_activo = False
    await _publicar({"tipo": "fin", **(estado.resumen or {})})
    log.step("Batch completo")


@app.get("/")
async def index() -> FileResponse:
    html = WEB / "index.html"
    if not html.exists():
        raise HTTPException(404, "Frontend no encontrado en image_processing/canada/web/index.html")
    return FileResponse(html)


@app.get("/api/muestras")
async def listar_muestras() -> dict:
    rutas = _muestras_iniciales()
    return {
        "muestras": [
            {"nombre": r.stem, "archivo": r.name, "estado": estado.muestras.get(r.stem, {}).get("estado", "pendiente")}
            for r in rutas
        ],
        "total": len(rutas),
    }


@app.post("/api/start")
async def iniciar() -> dict:
    async with estado.lock:
        if estado.procesando_activo:
            return {"ok": True, "ya_corriendo": True}
        if estado.terminado:
            return {"ok": True, "terminado": True}
        estado.procesando_activo = True
        asyncio.create_task(_correr_batch())
    return {"ok": True, "iniciado": True}


@app.get("/api/muestra/{nombre}")
async def detalle(nombre: str) -> dict:
    info = estado.muestras.get(nombre)
    if not info:
        raise HTTPException(404, "Muestra desconocida o aún no procesada")
    return {
        "nombre": nombre,
        **info,
        "rgb": f"/data/{nombre}.png",
        "overlay": f"/out/batch/{nombre}_overlay.png",
        "mascara": f"/out/batch/{nombre}_mascara.png",
        "hongo": f"/out/batch/{nombre}_hongo.png",
    }


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(_snapshot())
    estado.suscriptores.add(queue)

    async def gen():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            estado.suscriptores.discard(queue)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


WEB.mkdir(parents=True, exist_ok=True)
OUT_BATCH.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/out", StaticFiles(directory=OUT), name="out")
app.mount("/web", StaticFiles(directory=WEB), name="web")
