// Mildiu Scan — frontend
// Contrato del backend (image_processing/canada/app.py):
//   GET  /api/muestras   -> {muestras: [{nombre, archivo, estado}], total}
//   POST /api/start      -> idempotente; arranca el batch si no corre
//   GET  /api/stream     -> SSE con mensajes `data:` (todos via onmessage):
//                            {tipo:"snapshot", muestras:{nombre:{estado,pct?,...}}, procesando_activo, terminado, resumen}
//                            {tipo:"estado",   nombre, estado, pct?, area_hoja?, area_hongo?, mensaje?}
//                            {tipo:"fin",      promedio, min, max, n, csv}
//                            {tipo:"shutdown"}
//   Imágenes (StaticFiles):
//     /data/<nombre>.png             original
//     /out/batch/<nombre>_overlay.png
//     /out/batch/<nombre>_mascara.png
//     /out/batch/<nombre>_hongo.png

const TOTAL = 47;
const VIEW_PATH = {
  overlay: (n) => `/out/batch/${n}_overlay.png`,
  rgb:     (n) => `/data/${n}.png`,
  mascara: (n) => `/out/batch/${n}_mascara.png`,
  hongo:   (n) => `/out/batch/${n}_hongo.png`,
};

const $ = (sel, root = document) => root.querySelector(sel);
const fmt1 = (n) => (Math.round(n * 10) / 10).toFixed(1);
const fmt2 = (n) => (Math.round(n * 100) / 100).toFixed(2);
const fmtInt = (n) => n.toLocaleString("es-MX").replace(/,/g, " ");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const grid     = $("#grid");
const counter  = $("#counter");
const progress = $("#progress");
const metricEl = $("#metric");
const deltaEl  = $("#delta");
const rangeEl  = $("#range");
const caption  = $("#caption");
const pill     = $("#pill");
const pillLbl  = pill.querySelector(".pill__label");
const startBtn = $("#start");

const modal     = $("#modal");
const modalImg  = $("#modal-img");
const modalTitle= $("#modal-title");
const modalPct  = $("#modal-pct");
const modalLeaf = $("#modal-leaf");
const modalFung = $("#modal-fungus");
const modalFile = $("#modal-file");
const tabs      = $("#tabs");
const closeBtn  = $("#close");

const state = {
  cards: new Map(),           // nombre (string) -> { el, chip, num, data }
  pcts: new Array(TOTAL).fill(null),  // indexed por (parseInt(nombre)-1)
  done: 0,
  running: 0,
  chart: null,
  current: null,              // nombre de la muestra abierta
  metricaRaf: null,
  autoIniciado: false,
};

const idx = (nombre) => parseInt(nombre, 10) - 1;

// ------------------------------------------------------------
// Construcción inicial
// ------------------------------------------------------------

function construirGrid() {
  const tpl = $("#card-tpl");
  for (let i = 1; i <= TOTAL; i++) {
    const nombre = String(i);
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.dataset.nombre = nombre;
    node.querySelector(".card__num").textContent = String(i).padStart(2, "0");
    node.setAttribute("aria-label", `Muestra ${i}, pendiente`);
    node.addEventListener("click", () => abrirModal(nombre));
    grid.appendChild(node);
    state.cards.set(nombre, {
      el: node,
      chip: node.querySelector(".card__chip"),
      num: node.querySelector(".card__num"),
      data: { estado: "pendiente", archivo: `${nombre}.png` },
    });
  }
  for (let i = 0; i < TOTAL; i++) {
    const d = document.createElement("span");
    d.className = "progress__dot";
    progress.appendChild(d);
  }
}

function severidad(pct) {
  if (pct < 10) return "low";
  if (pct < 25) return "mid";
  return "high";
}

function pintarCard(nombre, payload) {
  const card = state.cards.get(nombre);
  if (!card) return;
  card.data = { ...card.data, ...payload };
  const { estado, pct } = card.data;
  const i = parseInt(nombre, 10);

  card.el.classList.remove("is-pending", "is-running", "is-done", "is-error");

  if (estado === "procesando") {
    card.el.classList.add("is-running");
    card.el.setAttribute("aria-label", `Muestra ${i}, procesando`);
    return;
  }
  if (estado === "error") {
    card.el.classList.add("is-error");
    card.chip.textContent = "!";
    card.chip.removeAttribute("data-sev");
    card.el.setAttribute("aria-label", `Muestra ${i}, error`);
    return;
  }
  if (estado === "listo") {
    card.el.classList.add("is-done");
    card.el.style.backgroundImage = `url(${VIEW_PATH.overlay(nombre)})`;
    card.chip.textContent = `${fmt1(pct)}%`;
    card.chip.dataset.sev = severidad(pct);
    card.el.setAttribute("aria-label", `Muestra ${i}, infección ${fmt1(pct)}%`);
    return;
  }
  card.el.classList.add("is-pending");
  card.el.setAttribute("aria-label", `Muestra ${i}, pendiente`);
}

// ------------------------------------------------------------
// Hero
// ------------------------------------------------------------

function animarMetrica(destino) {
  if (state.metricaRaf) cancelAnimationFrame(state.metricaRaf);
  if (reduceMotion) { metricEl.firstChild.textContent = fmt1(destino); return; }
  const desde = parseFloat(metricEl.firstChild.textContent) || 0;
  const t0 = performance.now();
  const dur = 700;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    metricEl.firstChild.textContent = fmt1(desde + (destino - desde) * eased);
    if (k < 1) state.metricaRaf = requestAnimationFrame(step);
    else state.metricaRaf = null;
  };
  state.metricaRaf = requestAnimationFrame(step);
}

function actualizarHero() {
  const done = state.pcts.filter((v) => v != null);
  if (done.length === 0) {
    metricEl.firstChild.textContent = "0.0";
    deltaEl.textContent = "— vs últimas 5";
    deltaEl.dataset.dir = "flat";
    rangeEl.textContent = "rango —";
    return;
  }
  const avg = done.reduce((a, b) => a + b, 0) / done.length;
  animarMetrica(avg);

  if (done.length >= 6) {
    const last5 = done.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const prev  = done.slice(0, -5);
    const prevAvg = prev.reduce((a, b) => a + b, 0) / prev.length;
    const diff = last5 - prevAvg;
    const dir = Math.abs(diff) < 0.05 ? "flat" : diff > 0 ? "up" : "down";
    const arrow = dir === "up" ? "↑" : dir === "down" ? "↓" : "=";
    deltaEl.textContent = `${arrow} ${fmt1(Math.abs(diff))} vs últimas 5`;
    deltaEl.dataset.dir = dir;
  } else {
    deltaEl.textContent = "— vs últimas 5";
    deltaEl.dataset.dir = "flat";
  }
  rangeEl.textContent = `rango ${fmt1(Math.min(...done))} % – ${fmt1(Math.max(...done))} %`;
}

function actualizarHeader() {
  counter.textContent = `${state.done} / ${TOTAL}`;
  progress.setAttribute("aria-valuenow", String(state.done));
  const dots = progress.querySelectorAll(".progress__dot");
  for (let i = 0; i < dots.length; i++) {
    dots[i].classList.toggle("is-done", i < state.done);
  }
}

function actualizarPill() {
  if (state.done >= TOTAL) {
    pill.dataset.state = "done";
    pillLbl.textContent = "Completado";
    caption.hidden = false;
    caption.textContent = `${TOTAL} muestras · listas`;
    startBtn.disabled = true;
    startBtn.textContent = "Lote completado";
    return;
  }
  if (state.running > 0) {
    pill.dataset.state = "running";
    pillLbl.textContent = `${state.running} procesando`;
    return;
  }
  pill.removeAttribute("data-state");
  pillLbl.textContent = state.done > 0 ? `${state.done} listas` : "En espera";
}

// ------------------------------------------------------------
// Chart
// ------------------------------------------------------------

function inicializarChart() {
  const ctx = $("#chart").getContext("2d");
  const labels = Array.from({ length: TOTAL }, (_, i) => i + 1);
  const gradient = ctx.createLinearGradient(0, 0, 0, 280);
  gradient.addColorStop(0, "rgba(10, 132, 255, 0.18)");
  gradient.addColorStop(1, "rgba(10, 132, 255, 0)");

  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data: new Array(TOTAL).fill(null),
        borderColor: "#0A84FF", borderWidth: 2,
        backgroundColor: gradient, fill: true, tension: 0.4,
        pointRadius: (c) => c.dataIndex === lastIndex(c.dataset.data) ? 4 : 0,
        pointHoverRadius: 5,
        pointBackgroundColor: "#0A84FF",
        pointBorderColor: "#fff", pointBorderWidth: 2,
        spanGaps: false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: reduceMotion ? false : { duration: 600, easing: "easeOutCubic" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(29, 29, 31, 0.92)", padding: 12, cornerRadius: 12,
          titleFont: { family: "-apple-system, system-ui", weight: "600", size: 13 },
          bodyFont:  { family: "-apple-system, system-ui", size: 13 },
          displayColors: false,
          callbacks: {
            title: (items) => `Muestra ${items[0].label}`,
            label: (item) => `${fmt2(item.parsed.y)} %`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false }, border: { display: false },
          ticks: { color: "#86868B", font: { family: "-apple-system, system-ui", size: 11 },
                   maxRotation: 0, autoSkip: true, autoSkipPadding: 16 },
        },
        y: {
          beginAtZero: true,
          grid: { color: "rgba(0,0,0,0.05)", drawTicks: false }, border: { display: false },
          ticks: { color: "#86868B", font: { family: "-apple-system, system-ui", size: 11 },
                   padding: 8, callback: (v) => `${v} %` },
        },
      },
    },
  });
}

function lastIndex(arr) {
  for (let i = arr.length - 1; i >= 0; i--) if (arr[i] != null) return i;
  return -1;
}

function pushChart(nombre, pct) {
  if (!state.chart) return;
  state.chart.data.datasets[0].data[idx(nombre)] = pct;
  state.chart.update();
}

// ------------------------------------------------------------
// Modal
// ------------------------------------------------------------

function abrirModal(nombre) {
  const card = state.cards.get(nombre);
  if (!card || !card.el.classList.contains("is-done")) return;
  state.current = nombre;
  const d = card.data || {};
  modalTitle.textContent = `Muestra ${nombre.padStart(2, "0")}`;
  modalPct.textContent   = d.pct != null ? `${fmt2(d.pct)} %` : "—";
  modalLeaf.textContent  = d.area_hoja != null ? fmtInt(d.area_hoja) : "—";
  modalFung.textContent  = d.area_hongo != null ? fmtInt(d.area_hongo) : "—";
  modalFile.textContent  = d.archivo || `${nombre}.png`;
  cambiarVista("overlay");
  if (!modal.open) modal.showModal();
}

function cambiarVista(view) {
  if (!state.current) return;
  const builder = VIEW_PATH[view];
  if (!builder) return;
  for (const t of tabs.querySelectorAll(".tab")) {
    const activo = t.dataset.view === view;
    t.classList.toggle("is-active", activo);
    t.setAttribute("aria-selected", activo ? "true" : "false");
  }
  const frame = modalImg.parentElement;
  frame.classList.add("is-swap");
  setTimeout(() => {
    modalImg.src = builder(state.current);
    modalImg.alt = `Muestra ${state.current} — vista ${view}`;
    requestAnimationFrame(() => frame.classList.remove("is-swap"));
  }, 160);
}

tabs.addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (btn) cambiarVista(btn.dataset.view);
});
closeBtn.addEventListener("click", () => modal.close());
modal.addEventListener("click", (e) => {
  const rect = modal.querySelector(".sheet").getBoundingClientRect();
  if (e.clientX < rect.left || e.clientX > rect.right ||
      e.clientY < rect.top  || e.clientY > rect.bottom) modal.close();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && modal.open) modal.close(); });

// ------------------------------------------------------------
// SSE — todos los mensajes llegan por onmessage, distinguidos por msg.tipo
// ------------------------------------------------------------

function aplicarEstado(nombre, payload) {
  const prev = (state.cards.get(nombre)?.data?.estado) || "pendiente";
  pintarCard(nombre, payload);
  const nuevo = payload.estado;

  if (nuevo === "procesando" && prev !== "procesando") state.running++;
  if ((nuevo === "listo" || nuevo === "error") && prev === "procesando") state.running--;
  if (nuevo === "listo" && prev !== "listo") {
    state.done++;
    if (typeof payload.pct === "number") {
      state.pcts[idx(nombre)] = payload.pct;
      pushChart(nombre, payload.pct);
    }
    actualizarHero();
  }
  actualizarHeader();
  actualizarPill();
}

function aplicarSnapshot(msg) {
  // muestras: dict { nombre: {estado, pct?, area_hoja?, area_hongo?} }
  state.done = 0;
  state.running = 0;
  state.pcts = new Array(TOTAL).fill(null);
  for (const [nombre, info] of Object.entries(msg.muestras || {})) {
    pintarCard(nombre, info);
    if (info.estado === "procesando") state.running++;
    if (info.estado === "listo") {
      state.done++;
      if (typeof info.pct === "number") {
        state.pcts[idx(nombre)] = info.pct;
        state.chart && (state.chart.data.datasets[0].data[idx(nombre)] = info.pct);
      }
    }
  }
  state.chart && state.chart.update();
  actualizarHeader();
  actualizarHero();
  actualizarPill();

  // Auto-arranque: si no está corriendo ni terminado, kickea el batch
  if (!msg.procesando_activo && !msg.terminado && !state.autoIniciado) {
    state.autoIniciado = true;
    iniciarBatch();
  } else if (msg.terminado) {
    finalizar();
  } else if (msg.procesando_activo) {
    startBtn.disabled = true;
    startBtn.textContent = "Procesando…";
  }
}

function finalizar() {
  pill.dataset.state = "done";
  pillLbl.textContent = "Completado";
  caption.hidden = false;
  caption.textContent = `${TOTAL} muestras · listas`;
  startBtn.disabled = true;
  startBtn.textContent = "Lote completado";
}

function conectarSSE() {
  const es = new EventSource("/api/stream");
  es.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    switch (msg.tipo) {
      case "snapshot": aplicarSnapshot(msg); break;
      case "estado":   aplicarEstado(msg.nombre, msg); break;
      case "fin":      finalizar(); break;
      case "shutdown": break;
    }
  };
  es.onerror = () => {
    if (state.done < TOTAL) {
      pill.removeAttribute("data-state");
      pillLbl.textContent = "Reconectando…";
    }
  };
}

// ------------------------------------------------------------
// Start (auto + click manual de respaldo)
// ------------------------------------------------------------

async function iniciarBatch() {
  startBtn.disabled = true;
  startBtn.textContent = "Procesando…";
  try {
    const r = await fetch("/api/start", { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (err) {
    console.error(err);
    startBtn.disabled = false;
    startBtn.textContent = "Reintentar";
  }
}

startBtn.addEventListener("click", iniciarBatch);

// ------------------------------------------------------------
// Boot
// ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  construirGrid();
  inicializarChart();

  try {
    const r = await fetch("/api/muestras");
    if (r.ok) {
      const resp = await r.json();
      for (const m of resp.muestras || []) {
        pintarCard(m.nombre, { estado: m.estado, archivo: m.archivo });
      }
    }
  } catch (err) {
    console.warn("GET /api/muestras falló, esperando SSE", err);
  }

  conectarSSE();
});
