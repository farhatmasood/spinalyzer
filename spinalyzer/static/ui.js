"use strict";

const $  = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const S = {
  studyId: null,
  data: null,
  files: [],          // {file, plane, weighting}
  activeSeries: null,
  view: "overlay",
};

/* ============================ helpers ============================ */

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Minimal markdown: tables, bold, italic, code, lists, paragraphs. */
function md(src) {
  const lines = String(src || "").split("\n");
  const out = [];
  let i = 0;

  const inline = (t) => esc(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");

  while (i < lines.length) {
    const ln = lines[i];

    if (/^\s*\|.*\|\s*$/.test(ln) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(ln);
      i += 2;
      const body = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) body.push(cells(lines[i++]));
      out.push("<table><thead><tr>" + head.map((h) => `<th>${inline(h)}</th>`).join("") +
               "</tr></thead><tbody>" +
               body.map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") +
               "</tbody></table>");
      continue;
    }

    if (/^\s*[-*]\s+/.test(ln)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`);
        i++;
      }
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (!ln.trim()) { i++; continue; }
    out.push(`<p>${inline(ln)}</p>`);
    i++;
  }
  return out.join("");
}

function size(px, mm, sq) {
  const u = sq ? "px&sup2;" : "px";
  if (mm == null) return `${px} ${u}`;
  return `${px} ${u} <span style="color:var(--ink-3)">/ ${mm} mm${sq ? "&sup2;" : ""}</span>`;
}

async function post(url, body) {
  const r = await fetch(url, { method: "POST", body });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* keep statusText */ }
    throw new Error(detail);
  }
  return r.json();
}

/* ============================ boot ============================ */

async function boot() {
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    const on = Object.entries(h.models).filter(([, v]) => v).map(([k]) => k);
    const chip = $("#status");
    chip.textContent = `${h.device} · ${on.length ? on.join(" · ") : "no model loaded"}`;
    chip.className = "chip " + (on.length === 3 ? "ok" : "bad");
  } catch (e) {
    $("#status").textContent = "backend offline";
    $("#status").className = "chip bad";
  }
  refreshLexicon();
}

/* ============================ file queue ============================ */

function addFiles(list) {
  for (const f of list) {
    if (S.files.length >= 8) break;
    S.files.push({ file: f, plane: "", weighting: "" });
  }
  renderQueue();
}

function renderQueue() {
  const t = $("#queue");
  const tb = t.querySelector("tbody");
  tb.innerHTML = "";
  t.hidden = S.files.length === 0;
  $("#analyse").disabled = S.files.length === 0;

  S.files.forEach((entry, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${esc(entry.file.name)}</td>
      <td>
        <select data-i="${idx}" data-k="plane">
          <option value="">auto</option>
          <option value="sagittal">sagittal</option>
          <option value="axial">axial</option>
        </select>
      </td>
      <td>
        <select data-i="${idx}" data-k="weighting">
          <option value="">auto</option>
          <option value="T1">T1</option>
          <option value="T2">T2</option>
        </select>
      </td>
      <td><button class="x" data-rm="${idx}" title="remove">&times;</button></td>`;
    tb.appendChild(tr);
    tr.querySelector('[data-k="plane"]').value = entry.plane;
    tr.querySelector('[data-k="weighting"]').value = entry.weighting;
  });

  tb.querySelectorAll("select").forEach((sel) => {
    sel.onchange = () => { S.files[+sel.dataset.i][sel.dataset.k] = sel.value; };
  });
  tb.querySelectorAll("[data-rm]").forEach((b) => {
    b.onclick = () => { S.files.splice(+b.dataset.rm, 1); renderQueue(); };
  });
}

$("#browse").onclick = () => $("#file").click();
$("#file").onchange = (e) => addFiles(e.target.files);

const drop = $("#drop");
drop.onclick = (e) => { if (e.target.id !== "browse") $("#file").click(); };
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("on"); };
drop.ondragleave = () => drop.classList.remove("on");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("on"); addFiles(e.dataTransfer.files); };

$("#load-samples").onclick = async () => {
  $("#upload-msg").textContent = "fetching samples…";
  const names = ["sag_T1_0001_S8.png", "axial_T1_0001_D4.png"];
  S.files = [];
  for (const n of names) {
    const r = await fetch(`/samples/${encodeURIComponent(n)}`);
    if (r.ok) S.files.push({ file: new File([await r.blob()], n), plane: "", weighting: "" });
  }
  $("#upload-msg").textContent = "";
  renderQueue();
};

/* ============================ analyse ============================ */

$("#analyse").onclick = async () => {
  if (!S.files.length) return;
  const btn = $("#analyse");
  btn.disabled = true;
  btn.textContent = "Analysing…";
  $("#upload-msg").textContent = "";

  const fd = new FormData();
  S.files.forEach((e) => fd.append("files", e.file, e.file.name));
  fd.append("planes", S.files.map((e) => e.plane).join(","));
  fd.append("weightings", S.files.map((e) => e.weighting).join(","));

  try {
    const d = await post("/api/study", fd);
    S.studyId = d.study_id;
    onStudy(d);
    $("#upload").hidden = true;
    $("#work").hidden = false;
    window.scrollTo(0, 0);
  } catch (err) {
    $("#upload-msg").textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyse study";
  }
};

/* ============================ render study ============================ */

function onStudy(d) {
  S.data = d;
  if (!S.activeSeries || !d.series.some((s) => s.id === S.activeSeries)) {
    S.activeSeries = d.reference_series_id || (d.series[0] && d.series[0].id);
  }
  renderStrip();
  renderView();
  renderAlign();
  renderGlance();
  renderAbnormal();
  renderTables();
  renderCalibration();

  if (d.warnings && d.warnings.length) {
    console.warn("study warnings", d.warnings);
  }
}

function activeSeries() {
  return S.data.series.find((s) => s.id === S.activeSeries) || S.data.series[0];
}

function renderStrip() {
  const el = $("#strip");
  el.innerHTML = "";
  S.data.series.forEach((s) => {
    const b = document.createElement("button");
    b.className = "thumb" + (s.id === S.activeSeries ? " is-on" : "");
    b.textContent = s.label;
    b.onclick = () => { S.activeSeries = s.id; renderStrip(); renderView(); renderTables(); renderCalibration(); };
    el.appendChild(b);
  });
}

function renderView() {
  const s = activeSeries();
  if (!s) return;
  $("#view").src = S.view === "overlay" ? s.overlay : s.original;
  const bits = [s.filename, s.format, `${s.shape[1]}×${s.shape[0]} px`];
  bits.push(s.mm_available ? `scale: ${s.spacing.source}` : "no pixel-size metadata");
  if (s.plane === "axial") bits.push(`level: ${s.level || "unassigned"} (${s.level_source})`);
  $("#view-meta").textContent = bits.join("  ·  ");
}

$$(".seg__b").forEach((b) => {
  b.onclick = () => {
    $$(".seg__b").forEach((x) => x.classList.remove("is-on"));
    b.classList.add("is-on");
    S.view = b.dataset.view;
    renderView();
  };
});

function renderAlign() {
  const rows = S.data.alignment || [];
  const tb = $("#align").querySelector("tbody");
  tb.innerHTML = "";
  $("#align").hidden = rows.length === 0;
  $("#align-empty").hidden = rows.length > 0;

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const opts = ["", ...S.data.assignable_levels]
      .map((l) => `<option value="${l}"${l === (r.level || "") ? " selected" : ""}>${l || "unassigned"}</option>`)
      .join("");
    tr.innerHTML = `
      <td>${esc(r.filename)}</td>
      <td>${esc(r.weighting)}</td>
      <td><select data-sid="${r.series_id}">${opts}</select></td>
      <td><span class="tag">${esc(r.level_source)}</span></td>`;
    tb.appendChild(tr);
  });

  tb.querySelectorAll("select").forEach((sel) => {
    sel.onchange = async () => {
      const fd = new FormData();
      fd.append("series_id", sel.dataset.sid);
      fd.append("level", sel.value);
      try { onStudy(await post(`/api/study/${S.studyId}/level`, fd)); }
      catch (e) { alert(e.message); }
    };
  });
}

function renderGlance() {
  const c = S.data.counts;
  const ref = S.data.series.find((s) => s.id === S.data.reference_series_id);
  const graded = ref ? ref.grading.length : 0;
  const cells = [
    [c.total, "images"],
    [c.sagittal, "sagittal"],
    [c.axial, "axial"],
    [S.data.disc_inventory.length, "disc levels"],
    [graded, "graded"],
    [S.data.abnormal_regions.length, "positive"],
  ];
  $("#glance").innerHTML = cells
    .map(([v, k]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
}

function renderAbnormal() {
  const el = $("#abnormal");
  const rs = S.data.abnormal_regions || [];
  if (!rs.length) {
    el.innerHTML = '<p class="empty">No level reached the 0.50 threshold.</p>';
    return;
  }
  el.innerHTML = rs.map((r) => {
    const m = r.metrics;
    const dims = m.height_mm != null
      ? `height ${m.height_mm} mm · AP ${m.ap_diameter_mm} mm`
      : `height ${m.height_px} px · AP ${m.ap_diameter_px} px`;
    const reasons = r.reasons
      .map((x) => x.kind + (x.score != null ? ` (${x.score.toFixed(2)})` : ""))
      .join(", ");
    return `<div class="abn__i">
        <div class="abn__l">${esc(r.level)}</div>
        <div class="abn__m">${esc(dims)}</div>
        <div class="abn__r">${esc(reasons)}</div>
      </div>`;
  }).join("");
}

function renderTables() {
  const s = activeSeries();
  const ref = S.data.series.find((x) => x.id === S.data.reference_series_id) || s;
  const abnormal = new Set(S.data.abnormal_regions.map((r) => r.level));

  $("#scale-tag").textContent = s.mm_available ? s.spacing.source : "pixels only";

  const dt = $("#t-disc").querySelector("tbody");
  dt.innerHTML = s.discs.length ? s.discs.map((m) => `
      <tr class="${abnormal.has(m.label) ? "flag" : ""}">
        <td>${esc(m.label)}</td>
        <td>${size(m.height_px, m.height_mm)}</td>
        <td>${size(m.ap_diameter_px, m.ap_diameter_mm)}</td>
        <td>${size(m.area_px, m.area_mm2, true)}</td>
      </tr>`).join("")
    : '<tr><td colspan="4" class="empty">No disc segmented.</td></tr>';

  const vt = $("#t-vb").querySelector("tbody");
  vt.innerHTML = s.vertebrae.length ? s.vertebrae.map((m) => `
      <tr><td>${esc(m.label)}</td>
        <td>${size(m.height_px, m.height_mm)}</td>
        <td>${size(m.width_px, m.width_mm)}</td>
        <td>${size(m.area_px, m.area_mm2, true)}</td></tr>`).join("")
    : '<tr><td colspan="4" class="empty">No vertebral body segmented (axial slices have none).</td></tr>';

  const gt = $("#t-grade").querySelector("tbody");
  gt.innerHTML = ref.grading.length ? ref.grading.map((g) => `
      <tr class="${g.pfirrmann_grade >= 3 || g.stenosis >= 1 ? "flag" : ""}">
        <td>${esc(g.level)}</td>
        <td><b>${esc(g.pfirrmann_roman)}</b></td>
        <td>${esc(g.pfirrmann_desc)}</td>
        <td>${esc(g.stenosis_label)} (${g.stenosis_probs[g.stenosis].toFixed(2)})</td>
      </tr>`).join("")
    : '<tr><td colspan="4" class="empty">No level graded. Grading covers L3-L4, L4-L5 and L5-S1 on a sagittal slice.</td></tr>';

  renderFindings(ref);
}

function renderFindings(ref) {
  const tb = $("#t-find").querySelector("tbody");
  const f = ref && ref.findings;
  if (!f || !f.binary_findings) {
    const why = (f && f.error) || "structured findings were not produced for this study";
    tb.innerHTML = `<tr><td colspan="3" class="empty">${esc(why)}.</td></tr>`;
    return;
  }
  const posOnly = $("#pos-only").checked;
  let items = f.binary_findings.slice().sort((a, b) => b.probability - a.probability);
  if (posOnly) items = items.filter((x) => x.positive);
  if (!items.length) {
    tb.innerHTML = '<tr><td colspan="4" class="empty">Nothing above the threshold.</td></tr>';
    return;
  }
  tb.innerHTML = items.map((x) => `
      <tr class="${x.positive && x.pathological ? "flag" : ""}">
        <td>${esc(x.name)}</td>
        <td>${x.probability.toFixed(3)}</td>
        <td>${x.positive ? "positive" : "negative"}</td>
        <td><button class="show" data-find="${esc(x.name)}"
             aria-pressed="false" title="mark this finding on the images">show</button></td>
      </tr>`).join("");

  tb.querySelectorAll(".show").forEach((b) => { b.onclick = () => toggleLocate(b); });
}

/* ---------------------- locate a named finding ---------------------- */

async function toggleLocate(btn) {
  const name = btn.dataset.find;
  const out = $("#locate-out");

  // Pressing the active button again clears the panel.
  if (btn.getAttribute("aria-pressed") === "true") {
    btn.setAttribute("aria-pressed", "false");
    btn.textContent = "show";
    out.innerHTML = "";
    return;
  }
  $$("#t-find .show").forEach((b) => {
    b.setAttribute("aria-pressed", "false");
    b.textContent = "show";
  });
  btn.setAttribute("aria-pressed", "true");
  btn.textContent = "hide";
  out.innerHTML = '<p class="hint">locating…</p>';

  const fd = new FormData();
  fd.append("finding", name);
  fd.append("threshold", String(parseFloat($("#thr").value)));
  try {
    const r = await post(`/api/study/${S.studyId}/locate`, fd);
    const L = r.locate || {};
    const head = `<p class="locate__h">${esc(L.finding || name)}</p>`;

    if (!r.images || !r.images.length) {
      out.innerHTML = head +
        `<p class="hint">${esc(L.reason || L.note || "nothing to mark for this finding.")}</p>`;
      return;
    }
    const ev = (L.evidence || []).map((e) =>
      `<tr><td>${esc(e.level || "-")}</td><td>${e.score != null ? e.score.toFixed(3) : "-"}</td>
           <td>${esc(e.source)}</td></tr>`).join("");
    out.innerHTML = head +
      (ev ? `<table class="tbl tbl--tiny"><thead><tr><th>Level</th><th>Score</th>
             <th>Source</th></tr></thead><tbody>${ev}</tbody></table>` : "") +
      r.images.map((im) =>
        `<figure><img src="${im.image}" alt="${esc(im.label)}">
         <figcaption>${esc(im.label)}</figcaption></figure>`).join("") +
      (L.note ? `<p class="hint">${esc(L.note)}</p>` : "");
  } catch (e) {
    out.innerHTML = `<p style="color:var(--flag)">${esc(e.message)}</p>`;
  }
}

$("#pos-only").onchange = () => {
  const ref = S.data.series.find((x) => x.id === S.data.reference_series_id);
  renderFindings(ref);
};

function renderCalibration() {
  const s = activeSeries();
  $("#cal-state").innerHTML = s.mm_available
    ? `Scale in use: <b>${esc(s.spacing.source)}</b> — ${s.spacing.row} × ${s.spacing.col} mm per pixel.`
    : "This file carries no pixel size, so measurements stay in pixels. Enter mm per pixel to convert the whole study.";
}

/* ============================ ask ============================ */

$("#ask-form").onsubmit = (e) => { e.preventDefault(); ask($("#ask").value.trim()); };
$$(".pill").forEach((b) => { b.onclick = () => ask(b.dataset.q); });

async function ask(text) {
  if (!text || !S.studyId) return;
  $("#ask").value = "";
  const box = $("#answer");
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `<div class="turn__q">${esc(text)}</div><div class="turn__a">working…</div>`;
  box.prepend(turn);

  const fd = new FormData();
  fd.append("text", text);
  try {
    const r = await post(`/api/study/${S.studyId}/ask`, fd);
    // A located finding can span several series, so render every image it returns.
    let pics = "";
    if (r.images && r.images.length) {
      pics = r.images.map((im) =>
        `<figure><img src="${im.image}" alt="${esc(im.label)}">
         <figcaption>${esc(im.label)}</figcaption></figure>`).join("");
    } else if (r.image) {
      pics = `<img src="${r.image}" alt="result">`;
    }
    turn.querySelector(".turn__a").innerHTML = md(r.text) + pics;
  } catch (err) {
    turn.querySelector(".turn__a").innerHTML = `<p style="color:var(--flag)">${esc(err.message)}</p>`;
  }
}

/* ============================ report ============================ */

function reportOptions() {
  const o = {};
  $$("#sections input, #layers input").forEach((c) => { o[c.dataset.o] = c.checked; });
  o.findings_threshold = parseFloat($("#thr").value);
  o.findings_positive_only = true;
  return o;
}

$("#thr").oninput = () => { $("#thr-out").textContent = parseFloat($("#thr").value).toFixed(2); };

$("#preview").onclick = async () => {
  const btn = $("#preview");
  btn.disabled = true; btn.textContent = "Rendering…";
  const fd = new FormData();
  fd.append("options", JSON.stringify(reportOptions()));
  try {
    const r = await post(`/api/study/${S.studyId}/report/preview`, fd);
    $("#preview-out").innerHTML = r.images.map((im) =>
      `<figure><img src="${im.image}" alt="${esc(im.label)}">
       <figcaption>${esc(im.label)}</figcaption></figure>`).join("");
  } catch (e) {
    $("#preview-out").innerHTML = `<p style="color:var(--flag)">${esc(e.message)}</p>`;
  } finally {
    btn.disabled = false; btn.textContent = "Preview images";
  }
};

$("#pdf").onclick = async () => {
  const btn = $("#pdf");
  btn.disabled = true; btn.textContent = "Building…";
  const fd = new FormData();
  fd.append("options", JSON.stringify(reportOptions()));
  try {
    const r = await fetch(`/api/study/${S.studyId}/report.pdf`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `spinalyzer_${S.studyId}.pdf`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    alert("PDF failed: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Download PDF";
  }
};

/* ============================ calibration ============================ */

$("#cal-form").onsubmit = async (e) => {
  e.preventDefault();
  const v = parseFloat($("#mmpx").value);
  if (!(v > 0)) return;
  const fd = new FormData();
  fd.append("mm_per_pixel", String(v));
  try { onStudy(await post(`/api/study/${S.studyId}/calibrate`, fd)); $("#mmpx").value = ""; }
  catch (err) { alert(err.message); }
};

/* ============================ vocabulary ============================ */

async function refreshLexicon() {
  try {
    const d = await fetch("/api/lexicon").then((r) => r.json());
    drawLexicon(d.aliases || []);
  } catch (e) { /* panel stays empty */ }
}

function drawLexicon(aliases) {
  const tb = $("#t-lex").querySelector("tbody");
  tb.innerHTML = aliases.length ? aliases.map((a) => `
      <tr><td><code>${esc(a.pattern)}</code></td>
          <td>${esc(a.expansion)}</td>
          <td><button class="x" data-p="${esc(a.pattern)}">&times;</button></td></tr>`).join("")
    : '<tr><td class="empty">No shortcut defined.</td></tr>';
  tb.querySelectorAll("[data-p]").forEach((b) => {
    b.onclick = async () => {
      const fd = new FormData();
      fd.append("text", `remove ${b.dataset.p}`);
      drawLexicon((await post("/api/lexicon", fd)).aliases || []);
    };
  });
}

$("#lex-form").onsubmit = async (e) => {
  e.preventDefault();
  const t = $("#lex").value.trim();
  if (!t) return;
  const fd = new FormData();
  fd.append("text", t);
  try {
    const r = await post("/api/lexicon", fd);
    drawLexicon(r.aliases || []);
    if (r.ok) $("#lex").value = "";
  } catch (err) { alert(err.message); }
};

/* ============================ reset ============================ */

$("#reset").onclick = () => {
  S.studyId = null; S.data = null; S.files = []; S.activeSeries = null;
  $("#answer").innerHTML = "";
  $("#preview-out").innerHTML = "";
  $("#file").value = "";
  renderQueue();
  $("#work").hidden = true;
  $("#upload").hidden = false;
  window.scrollTo(0, 0);
};

boot();
