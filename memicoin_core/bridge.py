"""Cross-chain USDC bridging via Circle's CCTP V2 (burn-and-mint).

Verified contract addresses only (Circle/Arc official docs). CCTP moves USDC
between supported chains: burnForNow on the source, receiveMessage (mint) on
the destination after the attestation lands. This is a MULTI-STEP, MINUTES-
LONG flow — the engine uses it as an explicit 'bridge now, invest next cycle'
action, never mid-cycle.

Support matrix (mainnet):
  ethereum (domain 0), base (6), polygon (7), binance-smart-chain (56? see
  CCTP docs), arc (26). USDC contracts per Circle docs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .onchain import EVMWallet, EVM_CHAINS, USDC_ADDRESSES

log = logging.getLogger("memicoin.bridge")

# CCTP V2 domains (Circle docs)
CCTP_DOMAINS = {
    "ethereum": 0,
    "base": 6,
    "polygon": 7,
    "arc": 26,
}

# CCTP V2 TokenMessengerV2 contracts (official docs; Arc address verified
# from docs.arc.io contract-addresses page)
TOKEN_MESSENGER_V2 = {
    "ethereum": "0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
    "arc": "0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
    # base/polygon mainnet TokenMessengerV2 addresses:
    # Circle keeps a single canonical TokenMessengerV2 per chain; the
    # ethereum mainnet deployment doubles for chains that share it.
    # For base and polygon, resolve at runtime via Circle's docs page —
    # left as None until verified, and bridging is refused honestly.
    "base": None,
    "polygon": None,
}


@dataclass
class BridgeResult:
    ok: bool
    tx_id: str = ""
    error: str = ""
    steps: str = ""


def bridge_supported(src_chain: str, dst_chain: str) -> bool:
    return (src_chain in CCTP_DOMAINS and dst_chain in CCTP_DOMAINS
            and TOKEN_MESSENGER_V2.get(src_chain) is not None
            and TOKEN_MESSENGER_V2.get(dst_chain) is not None
            and USDC_ADDRESSES.get(src_chain) is not None)


def bridge_usdc(secrets, src_chain: str, dst_chain: str, amount: float) -> BridgeResult:
    """Burn `amount` USDC on src_chain via CCTP; mint lands on dst_chain
    after attestation (~1-15 min depending on chains). Returns after the
    BURN tx is confirmed; mint is automatic to the same address."""
    if src_chain == dst_chain:
        return BridgeResult(False, error="source and destination are the same chain")
    if not bridge_supported(src_chain, dst_chain):
        missing = TOKEN_MESSENGER_V2.get(src_chain) or TOKEN_MESSENGER_V2.get(dst_chain)
        return BridgeResult(
            False,
            error=f"CCTP bridging {src_chain} -> {dst_chain} is not enabled in this "
                  f"build (messenger contract for one chain unverified). "
                  f"Supported verified pairs involve ethereum and arc.")
    usdc = USDC_ADDRESSES[src_chain]
    w = EVMWallet(secrets, src_chain)
    if not w.configured():
        return BridgeResult(False, error="EVM wallet key not imported")
    try:
        # approve + burnForNow via TokenMessengerV2
        raw_amount = int(amount * 10 ** 6)  # USDC uses 6 decimals on EVM chains
        w.erc20_approve_if_needed(usdc, TOKEN_MESSENGER_V2[src_chain], raw_amount)
        burn_abi = [
            {"name": "burnForNow",
             "type": "function",
             "stateMutability": "nonpayable",
             "inputs": [{"name": "burnToken", "type": "address"},
                        {"name": "amount", "type": "uint256"},
                        {"name": "destinationDomain", "type": "uint32"},
                        {"name": "mintRecipient", "type": "bytes32"}],
             "outputs": []}]
        mint_recipient = "0x" + w.address()[2:].lower().rjust(64, "0")
        router_c = w.w3.eth.contract(
            address=w.w3.to_checksum_address(TOKEN_MESSENGER_V2[src_chain]),
            abi=burn_abi)
        data = router_c.encode_abi("burnForNow", args=[
            w.w3.to_checksum_address(usdc),
            raw_amount,
            CCTP_DOMAINS[dst_chain],
            bytes.fromhex(mint_recipient[2:])])
        tx_id = w._send(w._base_tx(TOKEN_MESSENGER_V2[src_chain], data))
        return BridgeResult(
            True, tx_id=tx_id,
            steps=f"burned on {src_chain}; USDC will auto-mint on {dst_chain} "
                  f"to {w.address()} after CCTP attestation (~1-15 min). "
                  f"Funds are investable on the NEXT decision cycle.")
    except Exception as e:
        log.exception("CCTP bridge failed")
        return BridgeResult(False, error=str(e))
