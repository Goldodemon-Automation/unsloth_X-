// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

/* Operator console. Vanilla on purpose: the collector must run with no build step,
   no CDN and no asset pipeline, including on a host that only has Python. */

const WINDOW_CHOICES = [2, 4, 9, 12, 30, 180];
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const state = {
  days: 30,
  overview: null,
  installs: [],
  session: null,
  query: "",
  sort: { key: "last_seen", dir: -1 },
  chart: null,
};

const numberFormatter = new Intl.NumberFormat("en-US");

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? numberFormatter.format(n) : "0";
}

function compact(value) {
  const n = Number(value) || 0;
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(abs >= 1e13 ? 0 : 1)}T`;
  if (abs >= 1e9) return `${(n / 1e9).toFixed(abs >= 1e10 ? 0 : 1)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(abs >= 1e7 ? 0 : 1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(abs >= 1e4 ? 0 : 1)}K`;
  return num(n);
}

function pct(part, total) {
  if (!total) return "0%";
  return `${Math.round((Number(part) / Number(total)) * 100)}%`;
}

function plural(value, one, many) {
  return Number(value) === 1 ? one : many;
}

function shortModel(value) {
  const cleaned = String(value || "").replace(/\\/g, "/").replace(/\/+$/, "");
  const tail = cleaned.split("/").filter(Boolean).pop();
  return tail || cleaned || "unknown";
}

function stamp(value) {
  const epoch = Number(value);
  if (!Number.isFinite(epoch) || epoch <= 0) return "—";
  const date = new Date(epoch * 1000);
  const now = Date.now();
  const diff = Math.max(0, now - date.getTime());
  const minutes = Math.floor(diff / 60000);
  if (minutes < 2) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function dayLabel(value) {
  const day = String(value || "").slice(0, 10);
  const date = new Date(`${day}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return day;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { Accept: "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (response.status === 401) {
    const error = new Error("unauthorized");
    error.status = 401;
    throw error;
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = String(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return response.json();
}

/* ── Boot ─────────────────────────────────────────────────────────────── */

async function boot() {
  try {
    state.session = await api("/api/admin/session");
  } catch {
    state.session = { authenticated: false, tokenRequired: false };
  }
  if (!state.session.authenticated) {
    showLogin(state.session);
    return;
  }
  hideLogin();
  await load();
}

function showLogin(session) {
  const login = $("#login");
  login.hidden = false;
  const note = login.querySelector("p");
  const input = $("#login-token");
  const submit = login.querySelector("button[type=submit]");
  if (!session.tokenRequired) {
    note.textContent =
      "This collector has no admin token, so it only answers requests from the machine it runs on. Open it there, or set X8_TELEMETRY_ADMIN_TOKEN.";
    input.disabled = true;
    submit.disabled = true;
    submit.textContent = "Unavailable";
    return;
  }
  input.disabled = false;
  submit.disabled = false;
  submit.textContent = "Open console";
  input.focus();
}

function hideLogin() {
  $("#login").hidden = true;
  $("#logout").hidden = !state.session?.tokenRequired;
}

async function login(event) {
  event.preventDefault();
  const error = $("#login-error");
  error.hidden = true;
  try {
    await api("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: $("#login-token").value }),
    });
    state.session = await api("/api/admin/session");
    hideLogin();
    await load();
  } catch (failure) {
    error.textContent = failure.message === "unauthorized" ? "Wrong token." : failure.message;
    error.hidden = false;
  }
}

async function logout() {
  try {
    await api("/api/admin/logout", { method: "POST" });
  } catch {
    /* the cookie is gone either way once reloaded with a bad token */
  }
  window.location.reload();
}

/* ── Load ─────────────────────────────────────────────────────────────── */

async function load() {
  document.body.classList.add("is-loading");
  $("#sync-status").textContent = "Loading telemetry…";
  try {
    const [overview, installs] = await Promise.all([
      api(`/api/admin/overview?days=${state.days}`),
      api("/api/admin/installs"),
    ]);
    state.overview = overview;
    state.installs = installs.installs || [];
    render();
    document.body.classList.remove("is-loading", "is-stale");
    $("#rail-status").textContent = "Connected";
    $("#sync-status").textContent = `Updated ${new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}`;
    $("#sync-window").textContent = `${overview.since} → today · ${overview.days}d window`;
    $("#foot-meta").textContent = `${state.installs.length} installs on record · schema ${overview.generated_at ? "ok" : "—"}`;
  } catch (failure) {
    document.body.classList.remove("is-loading");
    document.body.classList.add("is-stale");
    $("#rail-status").textContent = "Disconnected";
    if (failure.status === 401) {
      showLogin({ tokenRequired: true });
      return;
    }
    $("#sync-status").textContent = `Could not load: ${failure.message}`;
  }
}

/* ── Render ───────────────────────────────────────────────────────────── */

function render() {
  const data = state.overview;
  if (!data) return;
  const hasInstalls = Number(data.totals.installs) > 0;
  $("#empty").hidden = hasInstalls;
  $("#content").hidden = !hasInstalls;
  if (!hasInstalls) {
    $("#sync-status").textContent = "Waiting for the first opt-in report";
    return;
  }

  renderKpis(data);
  renderWindows(data);
  renderChart(data);
  renderModels(data);
  renderTraining(data);
  renderMerges(data);
  renderDevices(data);
  renderFeatures(data);
  renderInstalls();
}

function renderKpis(data) {
  const totals = data.totals;
  const top = data.models[0];
  $("#kpi-installs").textContent = num(totals.active_installs);
  $("#kpi-installs-sub").textContent = `${num(totals.installs)} all time · ${num(totals.new_installs)} new`;
  $("#kpi-input").textContent = compact(totals.prompt_tokens);
  $("#kpi-input-sub").textContent = `${num(totals.requests)} responses · ${data.days}d`;
  $("#kpi-output").textContent = compact(totals.completion_tokens);
  const totalTokens = Number(totals.prompt_tokens) + Number(totals.completion_tokens);
  $("#kpi-output-sub").textContent = totalTokens
    ? `${pct(totals.completion_tokens, totalTokens)} of ${compact(totalTokens)} tokens`
    : "no tokens yet";
  $("#kpi-top-model").textContent = top ? shortModel(top.model) : "—";
  $("#kpi-top-model-sub").textContent = top
    ? `${num(top.uses)} ${plural(top.uses, "use", "uses")} across ${num(top.installs)} ${plural(top.installs, "install", "installs")}`
    : "no model usage";
  $("#kpi-trainings").textContent = num(totals.trainings);
  $("#kpi-trainings-sub").textContent = `${num(totals.completed_trainings)} completed`;
  $("#kpi-merges").textContent = num(totals.merges);
  $("#kpi-merges-sub").textContent = data.merges.total ? `${data.merges.methods.length} methods in play` : "nothing merged yet";
}

function renderWindows(data) {
  $("#windows").innerHTML = data.windows
    .map((window) => {
      const total = Number(window.prompt_tokens) + Number(window.completion_tokens);
      const values = data.series
        .slice(-window.days)
        .map((point) => Number(point.prompt_tokens) + Number(point.completion_tokens));
      return `
        <button type="button" class="window-card ${window.days === state.days ? "is-active" : ""}" data-days="${window.days}">
          <span class="w-day">${window.days} days</span>
          <span class="w-total">${compact(total)}</span>
          <span class="w-split">${compact(window.prompt_tokens)} in · ${compact(window.completion_tokens)} out</span>
          <span class="w-spark">${sparkBars(values)}</span>
        </button>`;
    })
    .join("");
  $$("#windows .window-card").forEach((button) => {
    button.addEventListener("click", () => setDays(Number(button.dataset.days)));
  });

  const switchHtml = WINDOW_CHOICES.map(
    (days) =>
      `<button type="button" data-days="${days}" class="${days === state.days ? "is-active" : ""}">${days}d</button>`,
  ).join("");
  $("#window-switch").innerHTML = switchHtml;
  $$("#window-switch button").forEach((button) => {
    button.addEventListener("click", () => setDays(Number(button.dataset.days)));
  });
}

function sparkBars(values) {
  const numbers = (values || []).map((value) => Number(value) || 0);
  // Downsample to a readable shape: 180 one-pixel bars in a 170px card would
  // force the card wider than its grid track and read as a texture anyway.
  const bucketCount = Math.min(28, numbers.length);
  const buckets = Array.from({ length: bucketCount }, () => 0);
  numbers.forEach((value, index) => {
    const bucket = Math.min(
      bucketCount - 1,
      Math.floor((index / Math.max(1, numbers.length)) * bucketCount),
    );
    buckets[bucket] += value;
  });
  const max = Math.max(1, ...buckets);
  return buckets
    .map((value) => {
      const height = Math.max(2, Math.round((value / max) * 18));
      return `<i style="height:${height}px${value ? "" : ";opacity:.35"}"></i>`;
    })
    .join("");
}

function setDays(days) {
  if (state.days === days) return;
  state.days = days;
  load();
}

/* ── Chart ────────────────────────────────────────────────────────────── */

function chartData(data) {
  const days = Math.min(state.days, 180);
  return data.series.slice(-days);
}

function renderChart(data) {
  const points = chartData(data);
  const svg = $("#chart-svg");
  const width = 1000;
  const height = 240;
  const pad = { top: 12, right: 8, bottom: 22, left: 44 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxInput = Math.max(1, ...points.map((p) => Number(p.prompt_tokens)));
  const maxOutput = Math.max(1, ...points.map((p) => Number(p.completion_tokens)));
  const max = Math.max(maxInput, maxOutput);
  const step = points.length > 1 ? plotW / (points.length - 1) : plotW;

  const x = (index) => pad.left + index * step;
  const yIn = (value) => pad.top + plotH - (Number(value) / max) * plotH;
  const yOut = (value) => pad.top + plotH - (Number(value) / max) * plotH;

  const linePath = (accessor, yFn) =>
    points.map((point, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${yFn(accessor(point)).toFixed(1)}`).join(" ");

  const areaPath = (accessor, yFn) =>
    `${linePath(accessor, yFn)} L${x(points.length - 1).toFixed(1)},${pad.top + plotH} L${pad.left},${pad.top + plotH} Z`;

  const grid = Array.from({ length: 5 }, (_, index) => {
    const value = (max / 4) * (4 - index);
    const y = pad.top + (plotH / 4) * index;
    return `<line class="grid" x1="${pad.left}" x2="${width - pad.right}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" />
      <text class="axis-label" x="4" y="${(y + 3).toFixed(1)}">${compact(value)}</text>`;
  }).join("");

  const labelCount = Math.min(7, points.length);
  const labels = Array.from({ length: labelCount }, (_, index) => {
    const pointIndex = Math.round((index * (points.length - 1)) / Math.max(1, labelCount - 1));
    return `<text class="axis-label" x="${x(pointIndex).toFixed(1)}" y="${height - 4}" text-anchor="middle">${esc(dayLabel(points[pointIndex]?.day))}</text>`;
  }).join("");

  svg.innerHTML = `
    <defs>
      <linearGradient id="grad-input" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#b8f36a" stop-opacity="0.34" />
        <stop offset="100%" stop-color="#b8f36a" stop-opacity="0.02" />
      </linearGradient>
      <linearGradient id="grad-output" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#5eead4" stop-opacity="0.3" />
        <stop offset="100%" stop-color="#5eead4" stop-opacity="0.02" />
      </linearGradient>
    </defs>
    ${grid}
    <path class="area-input" d="${areaPath((p) => p.prompt_tokens, yIn)}" />
    <path class="area-output" d="${areaPath((p) => p.completion_tokens, yOut)}" />
    <path class="line-input" d="${linePath((p) => p.prompt_tokens, yIn)}" />
    <path class="line-output" d="${linePath((p) => p.completion_tokens, yOut)}" />
    ${labels}
  `;

  state.chart = { points, pad, plotW, plotH, width, height, max, svg };
  bindChartHover();
}

function bindChartHover() {
  const svg = $("#chart-svg");
  const tip = $("#chart-tip");
  if (svg.dataset.bound === "1") return;
  svg.dataset.bound = "1";

  svg.addEventListener("mousemove", (event) => {
    const chart = state.chart;
    if (!chart) return;
    const bounds = svg.getBoundingClientRect();
    const px = ((event.clientX - bounds.left) / bounds.width) * chart.width;
    const ratio = Math.min(1, Math.max(0, (px - chart.pad.left) / chart.plotW));
    const index = Math.round(ratio * (chart.points.length - 1));
    const point = chart.points[index];
    if (!point) return;

    const cx = chart.pad.left + (chart.plotW / Math.max(1, chart.points.length - 1)) * index;
    const xPct = (cx / chart.width) * 100;
    const inY = chart.pad.top + chart.plotH - (Number(point.prompt_tokens) / chart.max) * chart.plotH;
    const outY = chart.pad.top + chart.plotH - (Number(point.completion_tokens) / chart.max) * chart.plotH;
    const yPct = Math.min(inY, outY) / chart.height;

    tip.hidden = false;
    tip.style.left = `${xPct}%`;
    tip.style.top = `${Math.max(10, yPct * 100 - 3)}%`;
    tip.innerHTML = `
      <div class="t-day">${esc(point.day)}</div>
      <div class="t-in">in ${num(point.prompt_tokens)}</div>
      <div class="t-out">out ${num(point.completion_tokens)}</div>
      <div style="color:#5e6c79">${num(point.requests)} responses · ${num(point.installs)} installs</div>`;
  });

  svg.addEventListener("mouseleave", () => {
    tip.hidden = true;
  });
}

/* ── Models ───────────────────────────────────────────────────────────── */

function renderModels(data) {
  const rows = data.models || [];
  const maxUses = Math.max(1, ...rows.map((row) => Number(row.uses)));
  const totalUses = rows.reduce((sum, row) => sum + Number(row.uses), 0);
  $("#model-count").textContent = `${rows.length} ${plural(rows.length, "model", "models")}`;
  $("#models-empty").hidden = rows.length > 0;
  $("#model-rows").innerHTML = rows
    .map((row, index) => {
      const kinds = (row.kinds || []).filter(Boolean);
      const tags = kinds.map((kind) => `<span class="tag ${kind === "api" ? "api" : "chat"}">${esc(kind)}</span>`).join("");
      return `
        <tr>
          <td class="rank"><b>${String(index + 1).padStart(2, "0")}</b></td>
          <td>
            <div class="model-cell">
              <span class="model-name" title="${esc(row.model)}">${esc(shortModel(row.model))}</span>
              ${tags}
            </div>
            <div style="color:#5e6c79;font-family:var(--mono);font-size:9.5px">${esc(row.model)}</div>
          </td>
          <td class="num">${num(row.uses)}</td>
          <td class="num">${num(row.installs)}</td>
          <td class="num">${compact(row.prompt_tokens)}</td>
          <td class="num">${compact(row.completion_tokens)}</td>
          <td>
            <div class="share">
              <div class="share-bar"><i style="width:${Math.max(2, (Number(row.uses) / maxUses) * 100)}%"></i></div>
              <span>${pct(row.uses, totalUses)}</span>
            </div>
          </td>
        </tr>`;
    })
    .join("");
}

/* ── Training ─────────────────────────────────────────────────────────── */

function barList(items, one = "install", many = "installs") {
  if (!items.length) return `<li class="feed-empty">No data in this window.</li>`;
  const max = Math.max(1, ...items.map((item) => Number(item.installs)));
  return items
    .map(
      (item) => `
      <li>
        <div class="bar-top">
          <span class="bar-name" title="${esc(item.name)}">${esc(shortModel(item.name))}</span>
          <span class="bar-value">${num(item.installs)} ${plural(item.installs, one, many)}</span>
        </div>
        <div class="bar-track"><i style="width:${Math.max(3, (Number(item.installs) / max) * 100)}%"></i></div>
      </li>`,
    )
    .join("");
}

function chipList(items) {
  if (!items.length) return `<li class="feed-empty">No data yet.</li>`;
  return items
    .map(
      (item) =>
        `<li>${esc(item.name)} <b>${num(item.installs)}</b></li>`,
    )
    .join("");
}

function renderTraining(data) {
  const training = data.trainings;
  const topBase = training.bases[0];
  $("#training-count").textContent = `${num(training.total)} ${plural(training.total, "run", "runs")}`;
  $("#training-top-base").textContent = topBase ? shortModel(topBase.name) : "—";
  $("#training-bases").innerHTML = barList(training.bases);
  $("#training-datasets").innerHTML = barList(training.datasets);
  $("#training-methods").innerHTML = chipList(training.methods);
  $("#training-status").innerHTML = chipList(training.status);
  $("#training-feed").innerHTML = training.recent.length
    ? training.recent
        .map((row) => {
          const status = String(row.status || "unknown").toLowerCase();
          const statusClass = status === "completed" ? "ok" : status === "error" ? "bad" : "";
          return `
            <div class="feed-row">
              <span class="f-day">${esc(dayLabel(row.day))}</span>
              <span class="f-main">${esc(shortModel(row.base_model) || "unknown base")}
                <small>on ${esc(shortModel(row.dataset) || "unknown dataset")} · ${num(row.steps)} steps</small>
              </span>
              <span class="f-tag ${statusClass}">${esc(row.method || "—")}</span>
            </div>`;
        })
        .join("")
    : `<div class="feed-empty">No training runs in this window.</div>`;
}

/* ── Merges ───────────────────────────────────────────────────────────── */

function renderMerges(data) {
  const merges = data.merges;
  const top = merges.parents[0];
  $("#merge-count").textContent = `${num(merges.total)} ${plural(merges.total, "merge", "merges")}`;
  $("#merge-top").textContent = top ? shortModel(top.name) : "—";
  $("#merge-parents").innerHTML = barList(merges.parents, "use", "uses");
  $("#merge-methods").innerHTML = chipList(merges.methods);
  $("#merge-feed").innerHTML = merges.recent.length
    ? merges.recent
        .map(
          (row) => `
          <div class="feed-row">
            <span class="f-day">${esc(dayLabel(row.day))}</span>
            <span class="f-main">${esc((row.models || []).map(shortModel).join(" + "))}
              <small>${esc((row.models || []).join(" + "))}</small>
            </span>
            <span class="f-tag">${esc(row.method || "—")}</span>
          </div>`,
        )
        .join("")
    : `<div class="feed-empty">No merges in this window.</div>`;
}

/* ── Devices ──────────────────────────────────────────────────────────── */

function renderDevices(data) {
  const devices = data.devices;
  $("#device-count").textContent = `${num(data.totals.active_installs)} active`;
  $("#device-gpus").innerHTML = barList(devices.gpus.slice(0, 6));
  $("#device-vram").innerHTML = barList(devices.vram.slice(0, 6));
  $("#device-ram").innerHTML = barList(devices.ram.slice(0, 6));
  $("#device-os").innerHTML = chipList(devices.os.slice(0, 6));
  $("#device-backends").innerHTML = chipList(devices.backends.slice(0, 6));
  $("#device-vendors").innerHTML = chipList(devices.vendors.slice(0, 6));
  $("#device-versions").innerHTML = chipList(devices.versions.slice(0, 6));
  $("#device-locales").innerHTML = chipList(devices.locales.slice(0, 6));
}

function renderFeatures(data) {
  const features = data.features || [];
  const max = Math.max(1, ...features.map((item) => Number(item.installs)));
  $("#features").innerHTML = features.length
    ? features
        .map(
          (item) => `
          <li>
            <div class="f-top"><span>${esc(item.label)}</span><b>${compact(item.value)}</b></div>
            <div class="bar-track"><i style="width:${Math.max(3, (Number(item.installs) / max) * 100)}%"></i></div>
            <span style="color:#5e6c79;font-size:9.5px">${num(item.installs)} ${plural(item.installs, "install uses", "installs use")} it</span>
          </li>`,
        )
        .join("")
    : `<li class="feed-empty">No feature counts reported yet.</li>`;
}

/* ── Installs ─────────────────────────────────────────────────────────── */

function filteredInstalls() {
  const needle = state.query.trim().toLowerCase();
  const rows = needle
    ? state.installs.filter((row) =>
        [
          row.install_id,
          row.version,
          row.os,
          row.gpu,
          row.gpu_vendor,
          row.cpu,
          row.top_model,
          row.locale,
        ]
          .filter(Boolean)
          .some((value) => String(value).toLowerCase().includes(needle)),
      )
    : state.installs.slice();

  const { key, dir } = state.sort;
  rows.sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    if (typeof av === "number" || typeof bv === "number") {
      return ((Number(av) || 0) - (Number(bv) || 0)) * dir;
    }
    return String(av || "").localeCompare(String(bv || "")) * dir;
  });
  return rows;
}

function renderInstalls() {
  const rows = filteredInstalls();
  $("#install-count").textContent = `${rows.length} of ${state.installs.length}`;
  $("#installs-empty").hidden = rows.length > 0;
  $("#install-rows").innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td title="${esc(new Date(Number(row.last_seen) * 1000).toISOString())}">${esc(stamp(row.last_seen))}</td>
        <td class="install-id" title="${esc(row.install_id)}">${esc(row.short_id)}</td>
        <td>${esc(row.version || "—")}</td>
        <td class="device-cell">
          ${esc(row.gpu || "Unknown GPU")}
          <small>${esc(row.os || "?")} · ${num(row.ram_gb)} GB RAM${row.vram_gb ? ` · ${num(row.vram_gb)} GB VRAM` : ""}</small>
        </td>
        <td class="num">${compact(row.tokens)}</td>
        <td class="num">${num(row.trainings)}</td>
        <td class="num">${num(row.merges)}</td>
        <td class="top-model-cell" title="${esc(row.top_model || "")}">${esc(shortModel(row.top_model) || "—")}</td>
      </tr>`,
    )
    .join("");

  $$(".sortable").forEach((header) => {
    header.classList.toggle("asc", state.sort.key === header.dataset.sort && state.sort.dir === 1);
    header.classList.toggle("desc", state.sort.key === header.dataset.sort && state.sort.dir === -1);
  });
}

/* ── Wire up ──────────────────────────────────────────────────────────── */

$("#refresh")?.addEventListener("click", () => load());
$("#logout")?.addEventListener("click", () => logout());
$("#login-form")?.addEventListener("submit", login);
$("#install-search")?.addEventListener("input", (event) => {
  state.query = event.target.value;
  renderInstalls();
});
$$(".sortable").forEach((header) => {
  header.addEventListener("click", () => {
    const key = header.dataset.sort;
    state.sort = {
      key,
      dir: state.sort.key === key ? -state.sort.dir : -1,
    };
    renderInstalls();
  });
});
$("#copy-endpoint")?.addEventListener("click", async () => {
  const url = `${window.location.origin}/api/v1/report`;
  try {
    await navigator.clipboard.writeText(url);
    $("#copy-endpoint").textContent = "Copied";
  } catch {
    $("#copy-endpoint").textContent = url;
  }
});
$("#empty-endpoint").textContent = `${window.location.origin}/api/v1/report`;

const observedSections = $$("section[id]");
const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    $$(".nav-link").forEach((link) => {
      link.classList.toggle("is-active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  },
  { rootMargin: "-20% 0px -65% 0px", threshold: [0.05, 0.2, 0.5] },
);
observedSections.forEach((section) => observer.observe(section));

boot();
