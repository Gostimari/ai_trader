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
- **Live mode (on-chain hot wallet):** import a private key and the app swaps directly on DEXs — Uniswap/PancakeSwap routers on Ethereum, Base, Polygon, BSC; Jupiter aggregator on Solana. The key is stored encrypted and used only for signing. In this mode the AI must provide token contract/mint addresses (bare tickers are rejected). TON is recognized but not enabled (its swap path could not be safely tested). Custodial brokers like Robinhood cannot be linked — there is no key to import. **Only import a dedicated hot wallet with limited funds.**
- **Secrets** (wallet keys, API keys) are encrypted at rest: OS keyring when available, otherwise a Fernet-encrypted local file with 0600 permissions. Secrets are never logged and never returned by the API.
- Hard constraints (max loss, allowed chains, max position size, profit target) are checked in code for every action, even in `full_control` mode (technical limits still apply).
- Emergency stop halts trading immediately and optionally converts holdings to a stablecoin.
- Every decision cycle stores the exact input sent to the AI and its full output (auditability).
- **Disclaimer:** meme coins are extremely volatile. This software does not guarantee profit.

## Testnet verification (do this before enabling mainnet)

The unit tests mock the blockchain RPC and Circle's attestation API, so they
prove the logic but not the live contract calls. Run ONE real bridge on
testnet to fully verify the on-chain + bridge stack with zero real funds:

1. **Get a throwaway key.** Create a fresh wallet (any wallet app) and import
   its private key in the dashboard (section 3). NEVER reuse a mainnet key.
2. **Get faucet tokens:**
   - Testnet USDC: [faucet.circle.com](https://faucet.circle.com) - claim on
     **Ethereum Sepolia** AND **Base Sepolia** (20 USDC per chain every 2h,
     no account needed).
   - Gas: Sepolia ETH from a public faucet (e.g. the Coinbase Developer
     Platform faucet dispenses both ETH and USDC on Sepolia and Base Sepolia).
3. **Configure the app (Settings):**
   - Enable **Testnet mode**.
   - Add `sepolia, base-sepolia` to **Allowed blockchains**.
   - Switch Trading Mode to **Real Money** (transactions are real, but on
     testnets - the funds are worthless faucet tokens).
4. **Run a decision cycle** (or propose a bridge action) moving USDC from
   `sepolia` to `base-sepolia`. The decision feed shows: burn tx on Sepolia
   -> attestation from Circle's sandbox Iris API -> mint tx on Base Sepolia.
   If the attestation is slow, the bridge is saved as pending and the mint
   completes automatically at the start of the next cycle.
5. Once the round-trip works (USDC arrives on Base Sepolia), disable Testnet
   mode for mainnet. Testnet chains are refused while OFF, mainnet chains are
   refused while ON - the two networks can never be mixed.

RPC endpoints can be overridden per chain via `MEMICOIN_RPC_<CHAIN>` (e.g.
`MEMICOIN_RPC_SEPOLIA`), and the CCTP attestation timeout via
`MEMICOIN_CCTP_TIMEOUT` (seconds, default 900).

## News & data

- The dashboard has a **Crypto news** section (CoinDesk / Cointelegraph / Decrypt RSS, no API key, 10-minute cache). If all feeds are unreachable the section says so instead of failing silently.
- **Settings · 5 · Data** exports a full JSON backup of your local state (settings, positions, decisions, cycles, demo wallet) via `/api/export`. Wallet keys are never exported — they stay in the encrypted store.
- The **Default on-chain chain** setting (Settings · 2) sets Mode B's home chain; `/api/go-live` also accepts it. Profit-target and max-loss stops work in demo and on-chain modes too: the app snapshots your wallet value (`pnl_baseline_usd`) at the first cycle after any wallet change and measures P&L against it.
- The daily scheduler retries a failed cycle every 30 minutes (AI down, market data down) instead of skipping the day, and a manual "Run now" can never overlap a scheduled cycle.

## Configuration

See the Settings section in the dashboard, or `.env` for API keys. All settings persist in the local SQLite database as versioned JSON.

## Porting notes

- Core package `memicoin_core` has zero platform-specific code.
- The web dashboard is plain HTML/JS with no build step — wraps cleanly in Tauri/Electron for desktop or a WebView for Android.
