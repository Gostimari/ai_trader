"""Automatic token discovery: what contracts does the linked wallet hold?

- Solana: token accounts are enumerable directly via public RPC
  (getTokenAccountsByOwner) — no key, no indexer.
- EVM: the chain itself cannot enumerate "tokens held"; the Etherscan-family
  V2 API (one free key, all Etherscan-family chains incl. Base/ArcScan)
  lists every ERC-20 the address has ever transacted. We collect the
  contract set from transfer history and read the ACTUAL balances on-chain
  afterwards (balanceOf) — the explorer only provides the candidate list.
"""
from __future__ import annotations

import logging

import requests

log = logging.getLogger("memicoin.discovery")

# Etherscan V2 multi-chain ids for the chains we support
ETHERSCAN_V2_CHAINIDS = {
    "ethereum": 1,
    "binance-smart-chain": 56,
    "polygon": 137,
    "base": 8453,
    "arc": 5042,
}

TIMEOUT = 20


class DiscoveryError(Exception):
    pass


def discover_evm_tokens(address: str, api_key: str,
                        chains: list[str] | None = None) -> dict[str, list[dict]]:
    """Return {chain: [{"address": "0x..", "symbol": "SYM", "decimals": n}]}
    for every ERC-20 contract the address has ever transacted, per chain.
    Chains not covered by the key (or with no history) are simply absent."""
    if not api_key:
        return {}
    out: dict[str, list[dict]] = {}
    for chain in (chains or list(ETHERSCAN_V2_CHAINIDS)):
        chainid = ETHERSCAN_V2_CHAINIDS.get(chain)
        if chainid is None:
            continue
        try:
            resp = requests.get(
                "https://api.etherscan.io/v2/api",
                params={
                    "chainid": chainid,
                    "module": "account",
                    "action": "tokentx",
                    "address": address,
                    "page": "1",
                    "offset": "300",
                    "sort": "desc",
                    "apikey": api_key,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            log.warning("etherscan v2 discovery failed for %s: %s", chain, e)
            continue
        if data.get("status") != "1" or not isinstance(data.get("result"), list):
            continue  # no history / unsupported chain: skip quietly
        seen: dict[str, dict] = {}
        for tx in data["result"]:
            contract = (tx.get("contractAddress") or "").lower()
            if not contract:
                continue
            if contract not in seen:
                seen[contract] = {
                    "address": contract,
                    "symbol": (tx.get("tokenSymbol") or "?")[:16],
                    "decimals": int(tx.get("tokenDecimal") or 18),
                }
        if seen:
            out[chain] = list(seen.values())
    return out


def discover_solana_tokens(owner_pubkey: str,
                           rpc_url: str = "https://api.mainnet-beta.solana.com") -> list[dict]:
    """All SPL tokens the wallet holds, via public RPC. No key needed.
    Returns [{"address": mint, "symbol": "?", "balance": uiAmount}] —
    symbols are resolved later via market data (mints are the identity)."""
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            owner_pubkey,
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss6BVi8Tp2gv"},
            {"encoding": "jsonParsed"},
        ],
    }
    resp = requests.post(rpc_url, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    result = (resp.json() or {}).get("result") or {}
    tokens = []
    for acc in result.get("value") or []:
        info = ((acc.get("account") or {}).get("data") or {}).get("parsed", {}).get("info", {})
        mint = info.get("mint") or ""
        amount = float((info.get("tokenAmount") or {}).get("uiAmount") or 0.0)
        if mint and amount > 0:
            tokens.append({"address": mint, "symbol": "?", "balance": amount})
    return tokens
