(function () {
  const $ = (s, c = document) => c.querySelector(s);

  function fmtBbox(b) {
    if (!b || b.length < 4) return "—";
    return b.map(n => n.toFixed(2)).join(",");
  }

  function fmtTimeAgo(ts) {
    const d = Math.max(0, Math.floor(Date.now() / 1000 - ts));
    if (d < 60) return `${d}s ago`;
    if (d < 3600) return `${Math.floor(d/60)}m ago`;
    return `${Math.floor(d/3600)}h ago`;
  }

  async function refresh() {
    let data;
    try { data = await fetch("/api/inference/events").then(r => r.json()); }
    catch { return; }
    const buf = $("#inference-buf");
    const rate = $("#inference-rate");
    const empty = $("#inference-empty");
    const table = $("#inference-table");
    const tbody = $("#inference-tbody");

    if (buf) buf.textContent = `events: ${data.buffer_size || 0}`;

    const counts = data.label_counts_60s || {};
    const labels = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    if (rate) {
      rate.textContent = labels.length
        ? labels.slice(0, 4).map(([l, n]) => `${l}: ${n}/min`).join(" · ")
        : "no recent events";
    }

    const items = data.items || [];
    if (!items.length) {
      empty.hidden = false;
      table.hidden = true;
      return;
    }
    empty.hidden = true;
    table.hidden = false;

    const rows = [];
    for (const ev of items) {
      const dets = ev.detections || [];
      if (!dets.length) {
        rows.push(`<tr><td>${fmtTimeAgo(ev.received_at)}</td><td>${esc(ev.path)}</td><td colspan=3 class="empty">no detections</td></tr>`);
      } else {
        for (const d of dets) {
          rows.push(`<tr>
            <td>${fmtTimeAgo(ev.received_at)}</td>
            <td>${esc(ev.path)}</td>
            <td>${esc(d.label)}</td>
            <td class="bytes">${(d.conf * 100).toFixed(1)}%</td>
            <td class="bytes">${esc(fmtBbox(d.bbox))}</td>
          </tr>`);
        }
      }
    }
    tbody.innerHTML = rows.join("");
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  document.addEventListener("DOMContentLoaded", () => {
    refresh();
    setInterval(refresh, 3000);
  });
})();
