// Inference plugin frontend — slice 1 wiring.
// Polls /api/inference/state to refresh the per-job status pill and the
// header backends badge. Settings/CRUD form lands in the next slice.
(function () {
  const $ = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => Array.from(c.querySelectorAll(s));

  function fmtStatus(j) {
    if (!j.enabled) return ["disabled", "idle"];
    return ["idle", "idle"];
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
      if (!j) {
        statusEl.textContent = "—";
        return;
      }
      const [text, kind] = fmtStatus(j);
      statusEl.textContent = text;
      statusEl.classList.remove("ok", "warn", "err", "idle");
      statusEl.classList.add(kind);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    // Wire the card-tab "Settings ▾" button to toggle the .settings-wrap.
    document.addEventListener("click", e => {
      const btn = e.target.closest("[data-inference-job] [data-act='settings']");
      if (!btn) return;
      const card = btn.closest("[data-inference-job]");
      const wrap = card.querySelector(".settings-wrap");
      if (!wrap) return;
      wrap.hidden = !wrap.hidden;
      btn.classList.toggle("open", !wrap.hidden);
    });

    refresh();
    setInterval(refresh, 5000);
  });
})();
