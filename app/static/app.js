const $ = (s) => document.querySelector(s);
const fmt = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d));
const pnlClass = (n) => (n > 0 ? "pos" : n < 0 ? "neg" : "");

async function api(path, body, method) {
  const opts = body ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function tabs() {
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab, .panel").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("#" + t.dataset.tab).classList.add("active");
    }));
}

async function loadState() {
  try {
    const s = await api("/api/state");
    $("#mode-badge").textContent = s.paper_trading ? "PAPER TRADING" : "LIVE TRADING";
    $("#mode-badge").classList.toggle("live", !s.paper_trading);
    $("#current-mode").textContent = s.paper_trading ? "PAPER" : "LIVE";
    $("#live-warning").classList.toggle("hidden", s.paper_trading);
    $("#btn-go-live").classList.toggle("hidden", !s.paper_trading);
    $("#btn-go-paper").classList.toggle("hidden", s.paper_trading);
    const p = s.portfolio;
    $("#portfolio-cards").innerHTML = `
      <div class="card"><div class="k">Total value</div><div class="v">$${fmt(p.total_value_usd)}</div></div>
      <div class="card"><div class="k">Cash (USD)</div><div class="v">$${fmt(p.cash_usd)}</div></div>
      <div class="card"><div class="k">Total P&L</div><div class="v ${pnlClass(p.total_pnl_usd)}">$${fmt(p.total_pnl_usd)}</div></div>
      <div class="card"><div class="k">Total P&L %</div><div class="v ${pnlClass(p.total_pnl_pct)}">${fmt(p.total_pnl_pct, 1)}%</div></div>`;
    $("#positions-table tbody").innerHTML = (p.positions || []).map((x) => `
      <tr><td>${x.asset}</td><td>${x.chain || ""}</td><td>${fmt(x.quantity, 4)}</td>
      <td>$${fmt(x.value_usd)}</td>
      <td class="${pnlClass(x.unrealized_pnl_usd)}">$${fmt(x.unrealized_pnl_usd)}</td>
      <td class="${pnlClass(x.unrealized_pnl_pct)}">${fmt(x.unrealized_pnl_pct, 1)}%</td></tr>`).join("")
      || `<tr><td colspan="6" class="hint">No positions yet. Run a decision cycle.</td></tr>`;
    const badges = [];
    if (s.trading_paused) badges.push("⏸ PAUSED");
    if (s.emergency_stopped) badges.push("🛑 EMERGENCY STOP");
    $("#status-badges").innerHTML = badges.map((b) => `<span class="badge live">${b}</span>`).join("");
    $("#btn-resume").classList.toggle("hidden", !(s.emergency_stopped || s.trading_paused));
    fillSettings(s.constraints);
  } catch (e) { console.error(e); }
}

function fillSettings(c) {
  $("#profit_target_pct").value = c.profit_target_pct ?? "";
  $("#max_loss_pct").value = c.max_loss_pct;
  $("#risk_profile").value = c.risk_profile;
  $("#max_position_pct").value = c.max_position_pct;
  $("#allowed_chains").value = (c.allowed_chains || []).join(", ");
  $("#decision_time_utc").value = c.decision_time_utc;
  $("#custom_prompt").value = c.custom_prompt || "";
  $("#autonomy_mode").value = c.autonomy_mode;
  $("#full-control-warning").classList.toggle("hidden", c.autonomy_mode !== "full_control");
}

async function loadDecisions() {
  const d = await api("/api/decisions");
  $("#decisions-table tbody").innerHTML = d.decisions.map((x) => `
    <tr><td>${x.timestamp?.slice(0, 19)}</td><td>${x.action}</td>
    <td>${x.asset_in || ""}${x.asset_out && x.asset_out !== "USD" ? " → " + x.asset_out : ""}</td>
    <td>${fmt(x.amount, 4)}</td><td>$${fmt(x.value_usd)}</td>
    <td class="${pnlClass(x.pnl_per_trade)}">$${fmt(x.pnl_per_trade)}</td>
    <td>${x.status}</td><td>${x.reason || ""}</td></tr>`).join("")
    || `<tr><td colspan="8" class="hint">No decisions yet.</td></tr>`;
}

async function loadNews() {
  const n = await api("/api/news");
  $("#news-list").innerHTML = n.news.map((x) => `
    <li>${x.url ? `<a href="${x.url}" target="_blank">${x.title}</a>` : x.title}
    <div class="src">${x.source}${x.published ? " · " + x.published.slice(0, 10) : ""}</div></li>`).join("");
}

function controls() {
  $("#btn-cycle").addEventListener("click", async () => {
    $("#btn-cycle").disabled = true;
    $("#cycle-result").textContent = "Running decision cycle…";
    try {
      const r = await api("/api/cycle");
      $("#cycle-result").textContent =
        `Cycle ${r.status.toUpperCase()}\n` +
        (r.analysis_summary ? `AI: ${r.analysis_summary} (confidence: ${r.confidence})\n` : "") +
        (r.error ? `Error: ${r.error}\n` : "") +
        (r.executed?.length ? `Executed ${r.executed.length} action(s)\n` : "") +
        (r.rejected?.length ? `Rejected ${r.rejected.length} action(s)` : "");
      await Promise.all([loadState(), loadDecisions()]);
    } catch (e) { $("#cycle-result").textContent = "Failed: " + e.message; }
    $("#btn-cycle").disabled = false;
  });
  $("#btn-pause").addEventListener("click", async () => { await api("/api/pause"); loadState(); });
  $("#btn-emergency").addEventListener("click", async () => {
    if (confirm("Halt all trading immediately?")) { await api("/api/emergency-stop"); loadState(); }
  });
  $("#btn-stable").addEventListener("click", async () => {
    if (confirm("Sell ALL positions and convert to cash? This cannot be undone.")) {
      await api("/api/emergency-to-stablecoin"); loadState(); loadDecisions();
    }
  });
  $("#btn-reset").addEventListener("click", async () => {
    if (confirm("Reset paper portfolio to starting capital?")) { await api("/api/reset-paper"); loadState(); loadDecisions(); }
  });
  $("#btn-resume").addEventListener("click", async () => {
    if (confirm("Re-enable trading after emergency stop/pause?")) { await api("/api/resume"); loadState(); }
  });
  $("#autonomy_mode").addEventListener("change", () =>
    $("#full-control-warning").classList.toggle("hidden", $("#autonomy_mode").value !== "full_control"));
  $("#settings-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const c = {
      profit_target_pct: $("#profit_target_pct").value === "" ? null : Number($("#profit_target_pct").value),
      max_loss_pct: Number($("#max_loss_pct").value),
      risk_profile: $("#risk_profile").value,
      max_position_pct: Number($("#max_position_pct").value),
      allowed_chains: $("#allowed_chains").value.split(",").map((s) => s.trim()).filter(Boolean),
      decision_time_utc: $("#decision_time_utc").value || "00:00",
      custom_prompt: $("#custom_prompt").value,
      autonomy_mode: $("#autonomy_mode").value,
    };
    await api("/api/constraints", c);
    alert("Settings saved. Constraints are enforced in code on every action.");
    loadState();
  });
}

async function loadLiveStatus() {
  try {
    const s = await api("/api/live-status");
    $("#secrets-status").textContent = s.configured
      ? `✅ Exchange "${s.exchange}" configured (stored in ${s.backend}).`
      : "No exchange credentials configured — live trading unavailable.";
  } catch (e) { $("#secrets-status").textContent = ""; }
}

function liveControls() {
  $("#secrets-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const body = {
      exchange_id: $("#exchange_id").value,
      api_key: $("#api_key").value,
      api_secret: $("#api_secret").value,
      api_password: $("#api_password").value,
    };
    if (!body.exchange_id) return alert("Enter an exchange id (e.g. binance).");
    const r = await api("/api/secrets", body);
    ["#api_key", "#api_secret", "#api_password"].forEach((s) => ($(s).value = ""));
    alert(`Credentials stored securely (${r.backend}). Reminder: use trade-only keys with withdrawals DISABLED.`);
    loadLiveStatus();
  });
  $("#btn-delete-secrets").addEventListener("click", async () => {
    if (confirm("Delete all stored exchange credentials?")) {
      await fetch("/api/secrets", { method: "DELETE" });
      loadLiveStatus();
    }
  });
  $("#btn-go-live").addEventListener("click", async () => {
    if (!confirm("Switch to LIVE trading with REAL funds?\n\nOnly continue if your API keys are trade-only with withdrawals disabled, and you have validated the AI in paper mode.")) return;
    if (!confirm("Final confirmation: REAL money will be traded. Are you absolutely sure?")) return;
    try {
      const r = await api("/api/go-live", { confirm: true });
      alert(`LIVE mode enabled on ${r.exchange}. The next decision cycle will trade real funds.`);
      loadState(); loadLiveStatus();
    } catch (e) { alert("Cannot go live: " + e.message); }
  });
  $("#btn-go-paper").addEventListener("click", async () => {
    if (!confirm("Switch back to PAPER trading?")) return;
    await api("/api/go-paper", {});
    loadState(); loadLiveStatus();
  });
}

async function main() {
  tabs();
  controls();
  liveControls();
  loadLiveStatus();
  await Promise.all([loadState(), loadDecisions(), loadNews()]);
  setInterval(loadState, 60000);
}
main();
