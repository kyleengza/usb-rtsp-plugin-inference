// Inference plugin frontend.
// Polls /api/inference/state for job status; lazily loads events + clips
// when their fold-out is opened.
(function () {
  const $ = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => Array.from(c.querySelectorAll(s));

  function esc(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function fmtAgo(ts) {
    const d = Math.max(0, Math.floor(Date.now() / 1000 - ts));
    if (d < 60) return `${d}s ago`;
    if (d < 3600) return `${Math.floor(d/60)}m ago`;
    if (d < 86400) return `${Math.floor(d/3600)}h ago`;
    return `${Math.floor(d/86400)}d ago`;
  }

  function fmtSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes/1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes/1024/1024).toFixed(1)} MB`;
    return `${(bytes/1024/1024/1024).toFixed(2)} GB`;
  }

  function fmtStatus(j) {
    if (!j.enabled) return ["disabled", "idle"];
    const live = j._live;
    if (!live) return ["idle", "idle"];
    if (live.ready) {
      const n = live.readers || 0;
      return [`${n} viewer${n === 1 ? "" : "s"}`, "ok"];
    }
    if ((live.readers || 0) > 0) return ["upstream unreachable", "err"];
    return ["idle (on-demand)", "idle"];
  }

  async function refresh() {
    let data;
    try {
      data = await fetch("/api/inference/state", { credentials: "same-origin" })
        .then(r => r.ok ? r.json() : null);
    } catch { return; }
    if (!data) return;

    const jobsByName = new Map((data.jobs || []).map(j => [j.name, j]));
    $$('article[data-inference-job]').forEach(card => {
      const name = card.dataset.inferenceJob;
      const j = jobsByName.get(name);
      const statusEl = card.querySelector("[data-status]");
      if (!statusEl) return;
      const [text, kind] = j ? fmtStatus(j) : ["—", "idle"];
      statusEl.textContent = text;
      statusEl.classList.remove("ok", "warn", "err", "idle");
      statusEl.classList.add(kind);
    });
  }

  async function loadEvents(card, name) {
    const tbody = card.querySelector("[data-events-table] tbody");
    if (!tbody) return;
    let events = [];
    try {
      const r = await fetch(`/api/inference/jobs/${encodeURIComponent(name)}/events?n=100`,
                           { credentials: "same-origin" });
      if (r.ok) events = (await r.json()).events || [];
    } catch { /* ignore */ }
    if (!events.length) {
      tbody.innerHTML = `<tr class="empty"><td colspan="5">no events recorded</td></tr>`;
      return;
    }
    // newest first
    events.reverse();
    tbody.innerHTML = events.map(ev => `<tr>
      <td>${esc(fmtAgo(ev.ts))}</td>
      <td>${esc(ev.kind)}</td>
      <td>#${ev.track_id}</td>
      <td>${esc(ev.label)}</td>
      <td>${(ev.score * 100).toFixed(1)}%</td>
    </tr>`).join("");
  }

  async function loadClips(card, name) {
    const tbody = card.querySelector("[data-clips-table] tbody");
    const totalEl = card.querySelector("[data-clips-total]");
    if (!tbody) return;
    let clips = [];
    let totalBytes = 0;
    try {
      const r = await fetch(`/api/inference/jobs/${encodeURIComponent(name)}/clips`,
                           { credentials: "same-origin" });
      if (r.ok) {
        const j = await r.json();
        clips = j.clips || [];
        totalBytes = j.total_size_bytes || 0;
      }
    } catch { /* ignore */ }
    if (totalEl) {
      totalEl.textContent = clips.length
        ? `${clips.length} clip${clips.length === 1 ? "" : "s"} · ${fmtSize(totalBytes)}`
        : "no clips";
    }
    if (!clips.length) {
      tbody.innerHTML = `<tr class="empty"><td colspan="4">no clips recorded</td></tr>`;
      return;
    }
    tbody.innerHTML = clips.map(c => {
      const dl = `/api/inference/clips/${encodeURIComponent(name)}/${encodeURIComponent(c.name)}`;
      return `<tr data-clip="${esc(c.name)}">
        <td>
          <button type="button" class="copy" data-clip-play="${esc(c.name)}">▶</button>
          <code>${esc(c.name)}</code>
        </td>
        <td>${esc(fmtSize(c.size_bytes))}</td>
        <td>${esc(fmtAgo(c.mtime))}</td>
        <td>
          <a href="${dl}" download="${esc(c.name)}">download</a>
          · <button type="button" class="copy" data-clip-delete="${esc(c.name)}">delete</button>
        </td>
      </tr>
      <tr class="clip-player-row" data-clip-player-for="${esc(c.name)}" hidden>
        <td colspan="4">
          <video controls preload="none" style="width:100%;max-width:720px;background:#000"
                 src="${dl}"></video>
        </td>
      </tr>`;
    }).join("");
  }

  async function deleteClip(name, file) {
    if (!confirm(`Delete clip ${file}?`)) return;
    try {
      const r = await fetch(`/api/inference/clips/${encodeURIComponent(name)}/${encodeURIComponent(file)}`,
                           { method: "DELETE", credentials: "same-origin" });
      if (!r.ok) alert(`delete failed: ${r.status}`);
    } catch (e) { alert(`delete error: ${e}`); }
  }

  function previewSrc(name) {
    return `/preview/${encodeURIComponent(name)}/`;
  }

  // Cache /api/inference/models so the model dropdown can be repopulated
  // when the backend toggle flips, without a roundtrip per click.
  let _modelsCache = null;
  async function getModels() {
    if (_modelsCache) return _modelsCache;
    try {
      const r = await fetch("/api/inference/models", { credentials: "same-origin" });
      if (!r.ok) return { hailo: [], cpu: [] };
      _modelsCache = await r.json();
      return _modelsCache;
    } catch { return { hailo: [], cpu: [] }; }
  }

  async function refreshModelOptions(form) {
    const backendInput = form.querySelector("input[name=backend]");
    const backend = backendInput ? backendInput.value : "hailo";
    const sel = form.querySelector("[name=model]");
    if (!sel) return;
    const current = sel.dataset.current || sel.value;
    const data = await getModels();
    const options = data[backend] || [];
    if (!options.length) {
      sel.innerHTML = `<option value="">(no ${backend} models on disk)</option>`;
      return;
    }
    sel.innerHTML = options.map(m => {
      const sel_attr = m.name === current ? " selected" : "";
      return `<option value="${m.name}"${sel_attr}>${m.name} · ${m.fps_target} fps target</option>`;
    }).join("");
  }

  function setStatus(form, kind, msg) {
    const el = form.querySelector("[data-form-status]");
    if (!el) return;
    el.textContent = msg;
    el.classList.remove("ok", "err");
    if (kind) el.classList.add(kind);
  }

  function selectedChips(form) {
    return $$("[data-class-chips] .class-chip.selected", form).map(b => b.dataset.class);
  }

  async function submitJob(form) {
    const fd = new FormData(form);
    const trig_classes_raw = String(fd.get("clip_trigger_classes") || "").trim();
    const payload = {
      name: fd.get("name"),
      upstream: fd.get("upstream"),
      enabled: true,  // toggled via the header switch, not this form
      backend: fd.get("backend"),
      model: fd.get("model"),
      classes: selectedChips(form),
      threshold: parseFloat(fd.get("threshold")) || 0.4,
      inference_queue: parseInt(fd.get("inference_queue") || "5", 10),
      track_occlusion_s: parseFloat(fd.get("track_occlusion_s")) || 2.0,
      clips: {
        enabled: !!fd.get("clips_enabled"),
        pre_roll_s: 0,  // v1: pre-roll not supported
        post_roll_s: parseInt(fd.get("clip_post_roll_s") || "10", 10),
        trigger: fd.get("clip_trigger") || "track_enter",
        trigger_classes: trig_classes_raw ? trig_classes_raw.split(",").map(s => s.trim()).filter(Boolean) : [],
        retention_count: parseInt(fd.get("clip_retention") || "100", 10),
      },
    };
    setStatus(form, null, "saving…");
    try {
      const r = await fetch(`/api/inference/jobs/${encodeURIComponent(payload.name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(payload),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setStatus(form, "err", data.detail || `HTTP ${r.status}`);
        return;
      }
      const bits = [`saved`, data.render || "", data.restart ? `restart=${data.restart}` : ""].filter(Boolean);
      setStatus(form, "ok", bits.join(" · "));
    } catch (e) {
      setStatus(form, "err", String(e));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    // Populate model dropdowns on page load + wire backend-pill + chips.
    $$("[data-inf-form]").forEach(form => {
      refreshModelOptions(form);
      const backendInput = form.querySelector("input[name=backend]");
      const backendPill = form.querySelector("[data-backend-pill]");
      if (backendPill) {
        backendPill.addEventListener("click", e => {
          const btn = e.target.closest("button[data-backend]");
          if (!btn || btn.classList.contains("active")) return;
          $$("button[data-backend]", backendPill).forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          if (backendInput) backendInput.value = btn.dataset.backend;
          const modelSel = form.querySelector("[name=model]");
          if (modelSel) modelSel.dataset.current = modelSel.value;  // remember last pick
          refreshModelOptions(form);
        });
      }
      // Class chip toggle.
      const chips = form.querySelector("[data-class-chips]");
      if (chips) {
        chips.addEventListener("click", e => {
          const chip = e.target.closest(".class-chip");
          if (!chip) return;
          chip.classList.toggle("selected");
        });
      }
      const clearBtn = form.querySelector("[data-chips-clear]");
      if (clearBtn) {
        clearBtn.addEventListener("click", () => {
          $$("[data-class-chips] .class-chip.selected", form).forEach(c => c.classList.remove("selected"));
        });
      }
      form.addEventListener("submit", e => {
        e.preventDefault();
        submitJob(form);
      });
    });

    document.addEventListener("click", e => {
      // Tab toggles inside an inference job card.
      const tab = e.target.closest("[data-inference-job] [data-act]");
      if (tab) {
        const card = tab.closest("[data-inference-job]");
        const act = tab.dataset.act;
        const name = card.dataset.inferenceJob;
        if (act === "preview-reconnect") {
          const iframe = card.querySelector("[data-preview-frame]");
          if (iframe) iframe.src = previewSrc(name) + "?t=" + Date.now();
          return;
        }
        const wrapMap = {
          preview: ".preview",
          urls: ".urls-wrap",
          events: ".events-wrap",
          clips: ".clips-wrap",
          settings: ".settings-wrap",
        };
        const wrapSel = wrapMap[act];
        if (!wrapSel) return;
        const wrap = card.querySelector(wrapSel);
        if (!wrap) return;
        wrap.hidden = !wrap.hidden;
        tab.classList.toggle("open", !wrap.hidden);
        if (!wrap.hidden) {
          if (act === "preview") {
            const iframe = card.querySelector("[data-preview-frame]");
            if (iframe && (!iframe.src || iframe.src.endsWith("about:blank"))) {
              iframe.src = previewSrc(name);
            }
          }
          if (act === "events") loadEvents(card, name);
          if (act === "clips") loadClips(card, name);
        } else {
          // Folding the preview closed must actually unload the iframe;
          // otherwise the WebRTC/HLS reader stays subscribed and mediamtx
          // never starts the runOnDemandCloseAfter countdown.
          if (act === "preview") {
            const iframe = card.querySelector("[data-preview-frame]");
            if (iframe) iframe.src = "about:blank";
          }
        }
        return;
      }
      // Per-clip delete button.
      const del = e.target.closest("[data-clip-delete]");
      if (del) {
        const card = del.closest("[data-inference-job]");
        const name = card?.dataset.inferenceJob;
        const file = del.dataset.clipDelete;
        if (name && file) deleteClip(name, file).then(() => loadClips(card, name));
        return;
      }
      // Per-clip play (inline video) toggle.
      const play = e.target.closest("[data-clip-play]");
      if (play) {
        const card = play.closest("[data-inference-job]");
        const file = play.dataset.clipPlay;
        const player = card.querySelector(`[data-clip-player-for="${CSS.escape(file)}"]`);
        if (player) {
          player.hidden = !player.hidden;
          play.classList.toggle("open", !player.hidden);
          const v = player.querySelector("video");
          if (v) {
            if (!player.hidden) v.play().catch(() => {});
            else v.pause();
          }
        }
      }
    });

    refresh();
    setInterval(refresh, 5000);
    // If any events/clips fold is open, refresh that one every 5s too.
    setInterval(() => {
      $$('article[data-inference-job]').forEach(card => {
        const name = card.dataset.inferenceJob;
        const evWrap = card.querySelector(".events-wrap");
        if (evWrap && !evWrap.hidden) loadEvents(card, name);
        const clWrap = card.querySelector(".clips-wrap");
        if (clWrap && !clWrap.hidden) loadClips(card, name);
      });
    }, 5000);
  });
})();
