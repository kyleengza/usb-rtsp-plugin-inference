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

  function fmtDuration(s) {
    if (s == null || isNaN(s) || s < 0) return "—";
    const total = Math.round(s);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
    return `${m}:${String(sec).padStart(2, "0")}`;
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

  function fmtPerf(j) {
    const s = j._stats;
    if (!s) return "";
    const bits = [];
    if (s.fps != null) bits.push(`${s.fps.toFixed(1)} fps`);
    if (s.inference_ms_avg != null) bits.push(`${s.inference_ms_avg.toFixed(1)} ms`);
    return bits.join(" · ");
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
      // Per-worker performance line (FPS + inference latency).
      const perfEl = card.querySelector("[data-perf]");
      if (perfEl) {
        const txt = j ? fmtPerf(j) : "";
        perfEl.textContent = txt;
        perfEl.hidden = !txt;
      }
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
      tbody.innerHTML = `<tr class="empty"><td colspan="5">no clips recorded</td></tr>`;
      return;
    }
    tbody.innerHTML = clips.map(c => {
      const dl = `/api/inference/clips/${encodeURIComponent(name)}/${encodeURIComponent(c.name)}`;
      return `<tr data-clip="${esc(c.name)}">
        <td>
          <button type="button" class="copy" data-clip-play="${esc(c.name)}">▶</button>
          <code>${esc(c.name)}</code>
        </td>
        <td>${esc(fmtDuration(c.duration_s))}</td>
        <td>${esc(fmtSize(c.size_bytes))}</td>
        <td>${esc(fmtAgo(c.mtime))}</td>
        <td>
          <a href="${dl}" download="${esc(c.name)}">download</a>
          · <button type="button" class="copy" data-clip-delete="${esc(c.name)}">delete</button>
        </td>
      </tr>
      <tr class="clip-player-row" data-clip-player-for="${esc(c.name)}" hidden>
        <td colspan="5">
          <video controls preload="none" style="width:100%;max-width:720px;background:#000"
                 src="${dl}"></video>
        </td>
      </tr>`;
    }).join("");
  }

  async function deleteClip(name, file) {
    if (!confirm(`Delete clip ${file}?`)) return false;
    try {
      const r = await fetch(`/api/inference/clips/${encodeURIComponent(name)}/${encodeURIComponent(file)}`,
                           { method: "DELETE", credentials: "same-origin" });
      if (!r.ok) {
        alert(`delete failed: ${r.status}`);
        return false;
      }
      return true;
    } catch (e) {
      alert(`delete error: ${e}`);
      return false;
    }
  }

  async function refreshClipsTotal(card, name) {
    // Lightweight: re-fetch only to update the totals line; doesn't touch rows.
    const totalEl = card.querySelector("[data-clips-total]");
    if (!totalEl) return;
    try {
      const r = await fetch(`/api/inference/jobs/${encodeURIComponent(name)}/clips`,
                           { credentials: "same-origin" });
      if (!r.ok) return;
      const j = await r.json();
      const n = (j.clips || []).length;
      totalEl.textContent = n
        ? `${n} clip${n === 1 ? "" : "s"} · ${fmtSize(j.total_size_bytes || 0)}`
        : "no clips";
    } catch { /* ignore */ }
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
      match_threshold: parseFloat(fd.get("match_threshold")) || 0.0,
      inference_queue: parseInt(fd.get("inference_queue") || "5", 10),
      track_occlusion_s: parseFloat(fd.get("track_occlusion_s")) || 2.0,
      min_hits: parseInt(fd.get("min_hits") || "3", 10),
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

  function fmtErrDetail(d, fallback) {
    if (d == null) return String(fallback);
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map(x => x?.msg || JSON.stringify(x)).join("; ");
    if (typeof d === "object") return d.msg || JSON.stringify(d);
    return String(d);
  }

  async function toggleAlwaysOn(card, name, pill) {
    if (pill.classList.contains("busy")) return;
    const wantOn = pill.classList.contains("off");
    pill.classList.add("busy");
    try {
      const r = await fetch(`/api/inference/jobs/${encodeURIComponent(name)}/always-on`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ enabled: wantOn }),
      });
      const text = await r.text();
      let data = null;
      try { data = JSON.parse(text); } catch {}
      if (!r.ok) {
        alert(`background toggle failed (HTTP ${r.status}): ${data ? fmtErrDetail(data.detail, r.status) : (text || r.status)}`);
        return;
      }
      pill.classList.toggle("on", wantOn);
      pill.classList.toggle("off", !wantOn);
      const strong = pill.querySelector("strong");
      if (strong) strong.textContent = wantOn ? "on" : "off";
    } catch (e) {
      alert(`background toggle error: ${e}`);
    } finally {
      pill.classList.remove("busy");
    }
  }

  async function toggleClips(card, name, pill) {
    if (pill.classList.contains("busy")) return;
    const wantOn = pill.classList.contains("off");
    pill.classList.add("busy");
    try {
      const r = await fetch(`/api/inference/jobs/${encodeURIComponent(name)}/clips`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ enabled: wantOn }),
      });
      const text = await r.text();
      let data = null;
      try { data = JSON.parse(text); } catch {}
      if (!r.ok) {
        const msg = data ? fmtErrDetail(data.detail, r.status) : (text || r.status);
        alert(`toggle failed (HTTP ${r.status}): ${msg}`);
        return;
      }
      pill.classList.toggle("on", wantOn);
      pill.classList.toggle("off", !wantOn);
      const strong = pill.querySelector("strong");
      if (strong) strong.textContent = wantOn ? "on" : "off";
    } catch (e) {
      alert(`toggle error: ${e}`);
    } finally {
      pill.classList.remove("busy");
    }
  }

  async function kickAllReadersForJob(name) {
    // Best-effort: when the user folds preview closed, mediamtx may take
    // up to runOnDemandCloseAfter to release the worker. Explicitly
    // kicking every webrtc/rtsp reader on the path makes the close
    // immediate. Failures are silent — this is purely a cleanup nudge.
    try {
      await fetch(`/api/inference/jobs/${encodeURIComponent(name)}/kick`, {
        method: "POST", credentials: "same-origin",
      });
    } catch { /* ignore */ }
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
          // Folding preview closed: src=about:blank on its own isn't
          // enough — Firefox's bfcache keeps the iframe's WebRTC client
          // alive in the background, and its built-in retry timer
          // re-fires WHEP every few seconds (visible as fresh
          // 127.0.0.1:<port> session creations in mediamtx logs even
          // after the worker has fully stopped). Replacing the iframe
          // element entirely severs that. Also kick mediamtx-side
          // readers so closeAfter starts immediately.
          if (act === "preview") {
            const iframe = card.querySelector("[data-preview-frame]");
            if (iframe) {
              const fresh = document.createElement("iframe");
              fresh.setAttribute("data-preview-frame", "");
              fresh.setAttribute("loading", "lazy");
              fresh.setAttribute("allow", "autoplay 'src'");
              fresh.setAttribute("allowfullscreen", "");
              iframe.replaceWith(fresh);
            }
            kickAllReadersForJob(name);
          }
          // Folding the Clips ▾ tab closed: tear down every inline clip
          // player so we stop pulling bytes and don't sit on multiple
          // mp4 readers in the background.
          if (act === "clips") {
            $$("[data-clip-player-for]", card).forEach(row => {
              row.hidden = true;
              const v = row.querySelector("video");
              if (v) {
                v.pause();
                v.removeAttribute("src");
                v.load();
              }
            });
            $$("[data-clip-play].open", card).forEach(b => b.classList.remove("open"));
          }
        }
        return;
      }
      // Quick background-mode on/off toggle in the meta row.
      const aoPill = e.target.closest("[data-always-on-toggle]");
      if (aoPill) {
        const card = aoPill.closest("[data-inference-job]");
        const name = card?.dataset.inferenceJob;
        if (name) toggleAlwaysOn(card, name, aoPill);
        return;
      }
      // Quick clip on/off toggle in the meta row.
      const clipsPill = e.target.closest("[data-clips-toggle]");
      if (clipsPill) {
        const card = clipsPill.closest("[data-inference-job]");
        const name = card?.dataset.inferenceJob;
        if (name) toggleClips(card, name, clipsPill);
        return;
      }
      // Per-clip delete button. Surgically remove just the deleted clip's
      // two rows so any inline <video> open elsewhere in the table keeps
      // playing — full loadClips() would tear down every row.
      const del = e.target.closest("[data-clip-delete]");
      if (del) {
        const card = del.closest("[data-inference-job]");
        const name = card?.dataset.inferenceJob;
        const file = del.dataset.clipDelete;
        if (!name || !file) return;
        deleteClip(name, file).then(ok => {
          if (!ok) return;
          card.querySelector(`tr[data-clip="${CSS.escape(file)}"]`)?.remove();
          card.querySelector(`tr[data-clip-player-for="${CSS.escape(file)}"]`)?.remove();
          // If the table is now empty, restore the empty placeholder row.
          const tbody = card.querySelector("[data-clips-table] tbody");
          if (tbody && !tbody.querySelector("tr[data-clip]")) {
            tbody.innerHTML = `<tr class="empty"><td colspan="4">no clips recorded</td></tr>`;
          }
          refreshClipsTotal(card, name);
        });
        return;
      }
      // Per-clip play (inline video) toggle.
      const play = e.target.closest("[data-clip-play]");
      if (play) {
        const card = play.closest("[data-inference-job]");
        const file = play.dataset.clipPlay;
        const player = card.querySelector(`[data-clip-player-for="${CSS.escape(file)}"]`);
        if (!player) return;
        const v = player.querySelector("video");
        if (player.hidden) {
          // Opening: re-attach src (rows are rendered with src already,
          // but a previous close may have stripped it). Then play.
          player.hidden = false;
          play.classList.add("open");
          if (v) {
            const dl = `/api/inference/clips/${encodeURIComponent(card.dataset.inferenceJob)}/${encodeURIComponent(file)}`;
            if (!v.src || v.src.endsWith("about:blank")) v.src = dl;
            v.play().catch(() => {});
          }
        } else {
          // Closing: pause AND release the resource so we stop pulling
          // bytes / holding a reader handle.
          player.hidden = true;
          play.classList.remove("open");
          if (v) {
            v.pause();
            v.removeAttribute("src");
            v.load();
          }
        }
      }
    });

    refresh();
    setInterval(refresh, 5000);
    // Events stream continuously while a worker is up — auto-refresh
    // those while the fold is open. Clips list only changes on
    // record/delete events that the user can trigger themselves; auto-
    // refresh of clips would rebuild the tbody and destroy any open
    // inline <video> player mid-playback. Refresh-on-open only.
    setInterval(() => {
      $$('article[data-inference-job]').forEach(card => {
        const name = card.dataset.inferenceJob;
        const evWrap = card.querySelector(".events-wrap");
        if (evWrap && !evWrap.hidden) loadEvents(card, name);
      });
    }, 5000);
  });
})();
