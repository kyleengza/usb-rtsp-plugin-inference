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
      let text = "—", kind = "idle";
      if (j) {
        text = j.enabled ? "idle" : "disabled";
        kind = "idle";
      }
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
    if (!tbody) return;
    let clips = [];
    try {
      const r = await fetch(`/api/inference/jobs/${encodeURIComponent(name)}/clips`,
                           { credentials: "same-origin" });
      if (r.ok) clips = (await r.json()).clips || [];
    } catch { /* ignore */ }
    if (!clips.length) {
      tbody.innerHTML = `<tr class="empty"><td colspan="4">no clips recorded</td></tr>`;
      return;
    }
    tbody.innerHTML = clips.map(c => `<tr>
      <td><a href="/api/inference/clips/${encodeURIComponent(name)}/${encodeURIComponent(c.name)}" target="_blank" rel="noopener">${esc(c.name)}</a></td>
      <td>${esc(fmtSize(c.size_bytes))}</td>
      <td>${esc(fmtAgo(c.mtime))}</td>
      <td><button type="button" class="copy" data-clip-delete="${esc(c.name)}">delete</button></td>
    </tr>`).join("");
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

  document.addEventListener("DOMContentLoaded", () => {
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
            if (iframe && !iframe.src) iframe.src = previewSrc(name);
          }
          if (act === "events") loadEvents(card, name);
          if (act === "clips") loadClips(card, name);
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
