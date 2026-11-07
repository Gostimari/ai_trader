#!/usr/bin/env python3
"""Audit the bot's on-chain transactions.

Reads the tx ids recorded in the app's SQLite database (plus any hash passed
on the command line), fetches each receipt from the chain RPC, and prints
exactly what happened: status, gas paid, ETH sent, and every ERC-20 token
transfer inside the transaction (contract, direction, amount, symbol).

Usage:
    python3 scripts/audit_txs.py            # audit everything in the DB
    python3 scripts/audit_txs.py 0xHASH...  # audit one specific transaction
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memicoin_core.db import DEFAULT_DB_PATH  # noqa: E402
from memicoin_core.onchain import EVM_CHAINS  # noqa: E402

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "symbol",
     "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals",
     "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
]


def _rpc(chain: str):
    from web3 import Web3
    rpc_url = os.environ.get(f"MEMICOIN_RPC_{chain.upper().replace('-', '_')}",
                             EVM_CHAINS[chain]["rpc"])
    return Web3(Web3.HTTPProvider(rpc_url))


def audit_tx(w3, tx_hash: str) -> None:
    print(f"\n=== tx {tx_hash}")
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        tx = w3.eth.get_transaction(tx_hash)
    except Exception as e:
        print(f"  not found on this chain ({e})")
        return
    gas_used = receipt["gasUsed"] / 1e18
    gas_price = tx["gasPrice"] / 1e9
    gas_eth = receipt["gasUsed"] * tx["gasPrice"] / 1e18
    value_eth = tx["value"] / 1e18
    print(f"  status : {'SUCCESS' if receipt['status'] == 1 else 'FAILED/REVERTED'}")
    print(f"  to     : {tx['to']}")
    print(f"  ETH    : value sent {value_eth:.6f}, gas paid {gas_eth:.8f} "
          f"({receipt['gasUsed']} @ {gas_price:.2f} gwei)")
    for lg in receipt["logs"]:
        topics = lg["topics"]
        topic0 = (topics[0].hex() if isinstance(topics[0], (bytes, bytearray))
                  else str(topics[0])) if topics else ""
        if topic0.startswith("0x"):
            topic0 = topic0[2:]
        if topic0 != TRANSFER_TOPIC[2:] or len(topics) < 3:
            continue
        token = lg["address"]
        try:
            c = w3.eth.contract(address=token, abi=ERC20_ABI)
            sym = c.functions.symbol().call()
            dec = c.functions.decimals().call()
        except Exception:
            sym, dec = "?", 18
        def _addr(t) -> str:
            h = t.hex() if isinstance(t, (bytes, bytearray)) else str(t)
            return "0x" + h[-40:]
        frm, to = _addr(topics[1]), _addr(topics[2])
        data = lg["data"]
        data_hex = data.hex() if isinstance(data, (bytes, bytearray)) else str(data)
        amount = int(data_hex, 16) / 10 ** dec
        print(f"  token  : {amount:,.6f} {sym} ({token})  {frm[:10]}.. -> {to[:10]}..")


def main() -> None:
    hashes = [a for a in sys.argv[1:] if a.startswith("0x") and len(a) == 66]
    db_hashes: list[str] = []
    db_path = os.environ.get("MEMICOIN_DB", DEFAULT_DB_PATH)
    if os.path.exists(db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT tx_id FROM decisions WHERE tx_id LIKE '0x%' "
                "AND status='executed' ORDER BY timestamp DESC").fetchall()
            db_hashes = [r[0] for r in rows if r[0]]
        finally:
            conn.close()
    todo = list(dict.fromkeys(hashes + db_hashes)) or []
    if not todo:
        print("No tx ids found (pass hashes as arguments, or run a decision "
              "cycle first). DB: " + db_path)
        return
    for chain in EVM_CHAINS:
        w3 = _rpc(chain)
        for h in todo:
            audit_tx(w3, h)


if __name__ == "__main__":
    main()
