const $ = (s) => document.querySelector(s);
const fmt = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d));
const pnlClass = (n) => (n > 0 ? "pos" : n < 0 ? "neg" : "");

async function api(path, body, method) {
  const opts = body !== undefined
    ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: method || "GET" };
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function tabs() {
  document.querySelectorAll("nav .tab").forEach((t) =>
    t.addEventListener("click", () => {
      if (!t.dataset.tab) return;
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
    $("#per_position_stop_pct").value = c.per_position_stop_pct ?? "";
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


function controls() {
  $("#btn-cycle").addEventListener("click", async () => {
    $("#btn-cycle").disabled = true;
    $("#cycle-result").textContent = "Running decision cycle…";
    try {
      const r = await api("/api/cycle", {});
      const fmtAction = (a) =>
        `${a.type.toUpperCase()} ${a.asset}${a.asset_out && a.asset_out !== "USD" ? " \u2192 " + a.asset_out : ""} \u00b7 ${a.amount_or_percent} \u00b7 "${a.reason}"`;
      $("#cycle-result").textContent =
        `Cycle ${r.status.toUpperCase()}\n` +
        (r.analysis_summary ? `AI: ${r.analysis_summary} (confidence: ${r.confidence})\n` : "") +
        (r.error ? `Error: ${r.error}\n` : "") +
        (r.executed?.length
          ? `\n\u2705 EXECUTED (${r.executed.length}):\n` + r.executed.map((a) => "  \u2022 " + fmtAction(a)).join("\n") + "\n"
          : "") +
        (r.rejected?.length
          ? `\n\u26d4 REJECTED by your constraints (${r.rejected.length}):\n` + r.rejected.map((a) => "  \u2022 " + fmtAction(a)).join("\n")
          : "") +
        (r.executed?.length || r.rejected?.length ? "" : "\nNo actions proposed (hold).");
      await Promise.all([loadState(), loadDecisions()]);
    } catch (e) { $("#cycle-result").textContent = "Failed: " + e.message; }
    $("#btn-cycle").disabled = false;
  });
  $("#btn-pause").addEventListener("click", async () => { await api("/api/pause", {}); loadState(); });
  $("#btn-emergency").addEventListener("click", async () => {
    if (confirm("Halt all trading immediately?")) { await api("/api/emergency-stop", {}); loadState(); }
  });
  $("#btn-stable").addEventListener("click", async () => {
    if (!confirm("Sell ALL positions and convert to USDC? This cannot be undone.")) return;
    const btn = $("#btn-stable");
    btn.disabled = true;
    btn.textContent = "Converting\u2026";
    try {
      const r = await api("/api/emergency-to-stablecoin", {});
      const msg = r.mode === "real"
        ? (r.ok ? "All liquidated to stablecoins."
                : "Liquidation finished with failures: " +
                  (r.results || []).filter((x) => !x.ok)
                    .map((x) => `${x.symbol} on ${x.chain}: ${x.error}`).join("; "))
        : "All paper positions converted to cash.";
      alert(msg);
    } catch (e) {
      alert("Liquidation FAILED: " + e.message);
    }
    btn.disabled = false;
    btn.textContent = "\ud83d\udcb5 Convert all to USDC (per chain)";
    loadState(); loadDecisions();
  });
  $("#btn-reset").addEventListener("click", async () => {
    if (confirm("Reset paper portfolio to starting capital?")) { await api("/api/reset-paper", {}); loadState(); loadDecisions(); }
  });
  $("#btn-resume").addEventListener("click", async () => {
    if (confirm("Re-enable trading after emergency stop/pause?")) { await api("/api/resume", {}); loadState(); }
  });
  $("#autonomy_mode").addEventListener("change", () =>
    $("#full-control-warning").classList.toggle("hidden", $("#autonomy_mode").value !== "full_control"));
  $("#settings-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const c = {
      profit_target_pct: $("#profit_target_pct").value === "" ? null : Number($("#profit_target_pct").value),
      max_loss_pct: Number($("#max_loss_pct").value),
      per_position_stop_pct: $("#per_position_stop_pct").value === "" ? null : Number($("#per_position_stop_pct").value),
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


function buildSeedInputs() {
  const wrap = $("#seed-words");
  if (!wrap || wrap.children.length) return;
  const fields = [];
  const make = (idx) => {
    const chip = document.createElement("span");
    chip.className = "seed-word";
    const num = document.createElement("span");
    num.textContent = idx + 1;
    num.style.color = "var(--muted)";
    num.style.fontSize = ".7rem";
    num.style.marginRight = "4px";
    const inp = document.createElement("input");
    inp.type = "password";
    inp.autocomplete = "off";
    inp.placeholder = "word";
    inp.addEventListener("input", () => {
      chip.classList.toggle("filled", inp.value.trim().length > 0);
      const { n, ok } = seedCount();
      updateSeedHint(n, ok);
    });
    inp.addEventListener("keydown", (e) => {
      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        const next = fields[idx + 1];
        if (next) next.focus();
      } else if (e.key === "Backspace" && !inp.value && idx > 0) {
        fields[idx - 1].focus();
      }
    });
    chip.appendChild(num);
    chip.appendChild(inp);
    fields.push(inp);
    return chip;
  };
  for (let i = 0; i < 24; i++) wrap.appendChild(make(i));
}

function seedPhrase() {
  return [...document.querySelectorAll("#seed-words input")]
    .map((i) => i.value.trim().toLowerCase()).filter(Boolean).join(" ");
}

function seedCount() {
  const n = seedPhrase() ? seedPhrase().split(" ").length : 0;
  return { n, ok: [12, 15, 18, 21, 24].includes(n) };
}

function updateSeedHint(n, ok) {
  const el = $("#seed-hint");
  if (!el) return;
  el.textContent = n === 0
    ? "Type your seed phrase \u2014 space or Enter jumps to the next word."
    : ok ? `${n} words \u2014 ready.`
    : `${n} word(s) \u2014 seed phrases are 12/15/18/21/24 words.`;
  el.style.color = ok ? "var(--green)" : "var(--muted)";
}

function importModeControls() {
  $("#btn-import-seed").addEventListener("click", () => {
    $("#btn-import-seed").classList.add("active");
    $("#btn-import-key").classList.remove("active");
    $("#seed-input-area").classList.remove("hidden");
    $("#key-input-area").classList.add("hidden");
  });
  $("#btn-import-key").addEventListener("click", () => {
    $("#btn-import-key").classList.add("active");
    $("#btn-import-seed").classList.remove("active");
    $("#key-input-area").classList.remove("hidden");
    $("#seed-input-area").classList.add("hidden");
  });
}
async function loadWalletStatus() {
  try {
    const s = await api("/api/wallet/status");
    const st = $("#wallet-status");
    if (s.evm_configured && s.solana_configured) {
      st.textContent = `✅ EVM wallet: ${s.evm_address} · Solana wallet: ${s.solana_address}`;
    } else if (s.evm_configured) {
      st.textContent = `✅ EVM multi-chain wallet linked: ${s.evm_address} (works on Ethereum, Base, Polygon, BSC, Arc)`;
      if (!s.solana_configured) st.textContent += " — no Solana key";
    } else if (s.solana_configured) {
      st.textContent = `✅ Solana wallet linked: ${s.solana_address} — no EVM key`;
    } else {
      st.textContent = "No wallet imported — real money mode needs at least one key.";
    }
  } catch (e) { $("#wallet-status").textContent = ""; }
}

function walletControls() {
  $("#wallet-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const type = $("#wallet-chain").value;
    const usingSeed = !$("#seed-input-area").classList.contains("hidden");
    const body = { chain: type, private_key: "", seed_phrase: "" };
    if (usingSeed) {
      const { n, ok } = seedCount();
      if (!ok) return alert(`Seed phrase has ${n} word(s) \u2014 must be 12/15/18/21/24.`);
      if (!confirm("Import this wallet for AUTONOMOUS trading?\n\nThe seed is stored encrypted and the AI will move these funds within your constraints.\n\nOnly the FIRST account (m/44'/60'/0'/0/0) is derived \u2014 if your funds live on a different account, use the raw key import instead.\n\nONLY use a dedicated hot wallet with limited funds.")) return;
      body.seed_phrase = seedPhrase();
    } else {
      const key = $("#wallet-key").value.trim();
      if (!key) return alert("Paste the wallet private key.");
      if (!confirm("Import this wallet for AUTONOMOUS trading?\n\nThe key is stored encrypted and the AI will move these funds within your constraints. ONLY use a dedicated hot wallet with limited funds.")) return;
      body.private_key = key;
    }
    try {
      const r = await api("/api/wallet/import", body);
      document.querySelectorAll("#seed-words input").forEach((i) => { i.value = ""; i.parentElement.classList.remove("filled"); });
      $("#wallet-key").value = "";
      updateSeedHint(0, false);
      alert(`Wallet linked${r.derived_from ? " (" + r.derived_from + ")" : ""}.\nVerify this address matches yours:\n${r.address}`);
      loadWalletStatus(); refreshModeUI();
    } catch (e) { alert("Import failed: " + e.message); }
  });
  $("#btn-wallet-delete").addEventListener("click", async () => {
    if (confirm("Remove ALL imported wallet keys from the app?")) {
      await fetch("/api/wallet", { method: "DELETE" });
      loadWalletStatus(); refreshModeUI();
    }
  });
}

async function refreshModeUI() {
  const s = await api("/api/state");
  const paper = s.paper_trading;
  $("#btn-mode-paper").classList.toggle("active", paper);
  $("#btn-mode-real").classList.toggle("active", !paper);
  $("#real-mode-warning").classList.toggle("hidden", paper);
  $("#mode-hint").textContent = paper
    ? "Paper mode: the AI trades against real market data with virtual funds. No risk."
    : "Real money mode: the AI trades your imported wallet's funds on DEXs. Constraints still enforced in code.";
}

function modeControls() {
  $("#btn-mode-paper").addEventListener("click", async () => {
    await api("/api/go-paper", {});
    refreshModeUI();
  });
  $("#btn-mode-real").addEventListener("click", async () => {
    const st = await api("/api/wallet/status");
    if (!st.evm_configured && !st.solana_configured)
      return alert("Import a wallet first (section 3 below).");
    if (!confirm("Switch to REAL MONEY mode?\n\nThe AI will trade your wallet's actual funds. Constraints still apply, but DEX swaps are irreversible.")) return;
    if (!confirm("Final confirmation: real funds will be traded. Continue?")) return;
    try { await api("/api/go-live", { confirm: true }); }
    catch (e) { return alert("Cannot switch: " + e.message); }
    refreshModeUI();
  });
}


function renderGasWarnings(gas) {
  const el = $("#gas-warning");
  if (!el) return;
  const warns = gas || [];
  if (!warns.length) { el.classList.add("hidden"); el.innerHTML = ""; return; }
  const worst = warns.some((w) => w.level === "empty" || w.level === "critical") ? "critical" : "low";
  el.className = "gas-warning " + worst;
  el.innerHTML = "\u26fd Gas reserve warnings:<br>" +
    warns.map((w) => `<div class="chain-line">\u2022 <b>${w.chain}</b>: ${w.hint}</div>`).join("");
}

async function checkGasWarnings() {
  try {
    const w = await api("/api/wallet/balances");
    renderGasWarnings(w.gas_warnings);
  } catch (e) { /* ignore */ }
}

async function loadWalletBalances() {
  const cards = $("#wallet-cards"), rowsEl = $("#wallet-balances"), hint = $("#wallet-hint");
  if (!cards) return;
  try {
    const w = await api("/api/wallet/balances");
    if (!w.configured) {
      cards.innerHTML = "";
      rowsEl.innerHTML = "";
      hint.textContent = "No wallet imported. Go to Settings \u00b7 3 Wallet to import your key.";
      return;
    }
    const all = Object.entries(w.chains).flatMap(([chain, rows]) =>
      rows.map((r) => ({ ...r, chain })));
    const total = w.total_usd;
    cards.innerHTML = `
      <div class="card"><div class="k">Wallet total</div><div class="v">$${fmt(total)}</div></div>
      <div class="card"><div class="k">Chains with funds</div><div class="v">${Object.keys(w.chains).length}</div></div>
      <div class="card"><div class="k">Tokens found</div><div class="v">${all.length}</div></div>`;
    hint.textContent = w.priced
      ? "Valued via DexScreener prices. Tokens with no DexScreener pair show as $0 \u2014 add their contract to the Watchlist to price them."
      : "Market data unavailable \u2014 balances shown without USD valuation.";
    // gas health strip
    const gas = w.gas || {};
    const gasRows = Object.entries(gas).map(([chain, g]) =>
      `<span class="gas-pill ${g.ok ? "ok" : "bad"}">${chain}: ${fmt(g.balance, 5)} ${g.symbol} \u00b7 ~${g.swaps_left} txs left \u00b7 investable ${fmt(g.investable ?? 0, 5)} ${g.ok ? "\u2705" : "\u26a0\ufe0f " + (g.hint || "low gas")}</span>`).join(" ");
    const gasEl = $("#gas-health");
    if (gasEl) gasEl.innerHTML = gasRows
      ? "<div class=\"gas-strip\">" + gasRows + "</div>" : "";
    if (!all.length) {
      rowsEl.innerHTML = "<p class='hint'>No balances found on any chain.</p>";
      return;
    }
    rowsEl.innerHTML = `<table><thead><tr><th>Chain</th><th>Token</th><th>Balance</th><th>Value $</th></tr></thead><tbody>` +
      all.sort((a, b) => b.value_usd - a.value_usd).map((r) => `
        <tr><td>${r.chain}</td><td>${r.symbol}${r.contract ? ` <span class="hint">${r.contract.slice(0, 8)}\u2026</span>` : ""}</td>
        <td>${fmt(r.balance, 6)}</td><td>$${fmt(r.value_usd)}</td></tr>`).join("") +
      `</tbody></table>`;
  } catch (e) {
    hint.textContent = "Failed to read wallet: " + e.message;
  }
}

function initWalletTab() {
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      if (t.dataset.tab === "wallet") loadWalletBalances();
    }));
}


let lastSeenCycleId = null;

async function pollCycles() {
  try {
    const cy = await api("/api/cycles/latest");
    if (!cy || !cy.id || cy.id === lastSeenCycleId) return;
    const isFirstLoad = lastSeenCycleId === null;
    lastSeenCycleId = cy.id;
    if (isFirstLoad) return; // don't notify for old cycles on page load
    const ok = cy.status === "completed";
    const msg = ok
      ? `Decision cycle finished: ${cy.analysis_summary || "see Decision Feed"}`
      : `Decision cycle ${cy.status}: ${cy.error || "check the Decision Feed"}`;
    const banner = $("#cycle-banner");
    if (banner) {
      banner.textContent = msg;
      banner.className = ok ? "banner ok" : "banner err";
      banner.classList.remove("hidden");
      setTimeout(() => banner.classList.add("hidden"), 30000);
    }
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification("MemeCoin AI Trader", { body: msg });
    }
    loadState(); loadDecisions();
  } catch (e) { /* silent */ }
}

function initCyclePolling() {
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }
  pollCycles();
  setInterval(pollCycles, 60000);
}


async function checkSetup() {
  try {
    const s = await api("/api/setup-status");
    const el = $("#setup-hint");
    if (!s.mistral_key) {
      el.innerHTML = "⚠️ No AI key configured: the app cannot make decisions. " +
        "Go to Settings \u00b7 1b AI Provider Key and paste your Mistral key (console.mistral.ai).";
      el.classList.remove("hidden");
    } else if (!s.wallet && !s.paper_trading) {
      el.textContent = "Real-money mode is on but no wallet is imported — trading will fail until you import a key in Settings.";
      el.classList.remove("hidden");
    }
  } catch (e) { /* ignore */ }
}


function mistralKeyControls() {
  $("#mistral-key-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const key = $("#mistral-key").value.trim();
    if (!key) return alert("Paste the API key first, or submit empty after clearing to remove it.");
    try {
      const r = await api("/api/mistral-key", { api_key: key });
      $("#mistral-key").value = "";
      $("#mistral-key-status").textContent = r.configured
        ? "\u2705 Key saved (encrypted). The AI can now run decision cycles."
        : "Key removed.";
      checkSetup();
    } catch (e) { alert("Failed: " + e.message); }
  });
}


function etherscanKeyControls() {
  $("#etherscan-key-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const key = $("#etherscan-key").value.trim();
    try {
      const r = await api("/api/etherscan-key", { api_key: key });
      $("#etherscan-key").value = "";
      $("#etherscan-key-status").textContent = key
        ? "\u2705 Key saved. Your EVM tokens will be discovered automatically."
        : "Key removed — only Watchlist tokens will be shown.";
    } catch (e) { alert("Failed: " + e.message); }
  });
}

async function loadWatchlist() {
  try {
    const w = await api("/api/watchlist");
    $("#watchlist-ids").value = (w.asset_ids || []).join(", ");
  } catch (e) {}
}

function watchlistControls() {
  $("#watchlist-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const ids = $("#watchlist-ids").value.split(",").map((s) => s.trim()).filter(Boolean);
    try {
      await api("/api/watchlist", { asset_ids: ids });
      $("#watchlist-status").textContent = `Saved. The AI will now consider ${ids.length} extra asset(s).`;
    } catch (e) { alert("Failed: " + e.message); }
  });
}

async function main() {
  tabs();
  controls();
  modeControls();
  mistralKeyControls();
  walletControls();
  buildSeedInputs();
  importModeControls();
  watchlistControls();
  etherscanKeyControls();
  loadWalletStatus();
  loadWatchlist();
  refreshModeUI();
  initWalletTab();
  initCyclePolling();
  checkSetup();
  await Promise.all([loadState(), loadDecisions(), loadNews()]);
  setInterval(loadState, 60000);
  checkGasWarnings();
  setInterval(checkGasWarnings, 120000);
}
main();
