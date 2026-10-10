# MemeCoin AI Trader

A desktop investment application (Linux-first, designed for porting to Windows/macOS/Android) that:

- Continuously ingests meme-coin market data (CoinGecko API, DexScreener fallback)
- Runs an AI (Mistral API) decision cycle once per day (configurable, manual trigger available)
- **Paper trading by default** ($100 USD virtual budget); optional **live trading** (Phase 2) on a real exchange via CCXT — explicit confirmation required, trade-only API keys
- Enforces all user constraints (max loss, allowed chains, max position size, autonomy mode) **deterministically in code** — the LLM only proposes, code validates and executes
- Shows full decision transparency: every action, amount, P&L, and a plain-language reason, including rejected proposals
- Includes a crypto news feed (CryptoPanic-style adapter with offline fallback)

## Architecture

All business logic lives in the platform-independent `memicoin_core` package (pure Python, no OS-specific dependencies). The UI is a thin layer: a FastAPI local server + static web dashboard (`app/`). No trading logic exists in the UI.

```
[Market Data Module] ──► [AI Decision Engine] ──► [Execution (Paper/Live)]
        │                        │                        │
        ▼                        ▼                        ▼
[News Module]            [SQLite Decision Log] ◄── [Portfolio & P&L Tracker]
                                 │
                                 ▼
                         [Web Dashboard UI]
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add MISTRAL_API_KEY
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in a browser.

Run tests:

```bash
pytest
```

## Safety

- **Paper trading by default.** The AI trades against real market data with virtual funds. Switching to live requires configuring exchange credentials and an explicit double-confirmed `/api/go-live` switch; the mode cannot be flipped through the normal settings save.
- **Live mode (Mode A, exchange):** trades via CCXT spot market orders. Use API keys with **trade-only permission, withdrawals disabled** — the app never calls withdrawal endpoints. Every order uses a deterministic client order ID (idempotent on retries) and balances are reconciled after each fill; a mismatch fails the action loudly.
- **Live mode (Mode B, on-chain hot wallet):** import a private key and the app swaps directly on DEXs — Uniswap/PancakeSwap routers on Ethereum, Base, Polygon, BSC; Jupiter aggregator on Solana. The key is stored encrypted and used only for signing. In this mode the AI must provide token contract/mint addresses (bare tickers are rejected). TON is recognized but not enabled (its swap path could not be safely tested). Custodial brokers like Robinhood cannot be linked — there is no key to import. **Only import a dedicated hot wallet with limited funds.**
- **Secrets** (exchange keys) are encrypted at rest: OS keyring when available, otherwise a Fernet-encrypted local file with 0600 permissions. Secrets are never logged and never returned by the API.
- Hard constraints (max loss, allowed chains, max position size, profit target) are checked in code for every action, even in `full_control` mode (technical limits still apply).
- Emergency stop halts trading immediately and optionally converts holdings to a stablecoin.
- Every decision cycle stores the exact input sent to the AI and its full output (auditability).
- **Disclaimer:** meme coins are extremely volatile. This software does not guarantee profit.

## Configuration

See the Settings section in the dashboard, or `.env` for API keys. All settings persist in the local SQLite database as versioned JSON.

## Porting notes

- Core package `memicoin_core` has zero platform-specific code.
- The web dashboard is plain HTML/JS with no build step — wraps cleanly in Tauri/Electron for desktop or a WebView for Android.
