"""Cross-chain USDC bridging via Circle's CCTP V2 (burn-and-mint).

Real-mode flow (verified against Circle's and Arc's official docs):
  1. approve USDC spend for the source chain's TokenMessengerV2
  2. depositForBurn(amount, destinationDomain, mintRecipient, burnToken,
     destinationCaller, maxFee, minFinalityThreshold) on TokenMessengerV2
  3. pull the MessageSent event log from the burn receipt; keccak256 of the
     message bytes is the message hash
  4. poll Circle Iris: GET /v2/attestations/{messageHash} until complete
  5. receiveMessage(message, attestation) on the destination's
     MessageTransmitterV2 -> USDC minted to the same address

CCTP is NOT fire-and-forget: steps 4 and 5 are mandatory, otherwise funds
stay burned-but-unminted. When the attestation does not land within the
timeout, the bridge is persisted (pending_bridges table) and completed
automatically on a later cycle by resume_pending_bridges().
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from .onchain import EVMWallet, USDC_ADDRESSES

log = logging.getLogger("memicoin.bridge")

# CCTP V2 domains (Circle docs). Testnet chains share the mainnet domain
# IDs (sepolia=ethereum's 0, base-sepolia=base's 6).
CCTP_DOMAINS = {
    "ethereum": 0,
    "base": 6,
    "polygon": 7,
    "arc": 26,
    "sepolia": 0,
    "base-sepolia": 6,
}

# Testnet chains: verify flows here BEFORE enabling mainnet. Cross-network
# bridges (mainnet <-> testnet) are refused - the USDC would never mint.
TESTNET_CHAINS = {"sepolia", "base-sepolia"}

# Circle deploys the SAME canonical V2 contracts on every chain of a given
# network (verified: ethereum 0, base 6, polygon 7, arc 26; sepolia +
# base-sepolia share the testnet deployments).
TOKEN_MESSENGER_V2 = "0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d"
MESSAGE_TRANSMITTER_V2 = "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64"
TOKEN_MESSENGER_V2_TESTNET = "0x8FE6B999Dc680CcFDD5Bf7EB0974218be2542DAA"
MESSAGE_TRANSMITTER_V2_TESTNET = "0xE737e5cEBEEBa77EFE34D4aa090756590b1CE275"

IRIS_MAINNET = "https://iris-api.circle.com"
IRIS_SANDBOX = "https://iris-api-sandbox.circle.com"


def is_testnet_chain(chain: str) -> bool:
    return chain in TESTNET_CHAINS


def messenger_for(chain: str) -> str:
    return TOKEN_MESSENGER_V2_TESTNET if is_testnet_chain(chain) else TOKEN_MESSENGER_V2


def transmitter_for(chain: str) -> str:
    return (MESSAGE_TRANSMITTER_V2_TESTNET if is_testnet_chain(chain)
            else MESSAGE_TRANSMITTER_V2)


def iris_base_for(chain: str) -> str:
    return IRIS_SANDBOX if is_testnet_chain(chain) else IRIS_MAINNET

# Arc quirk: USDC is the native gas token there, so CCTP burns a dedicated
# burn-token contract instead of an ERC-20 USDC address (docs.arc.io).
ARC_CCTP_BURN_TOKEN = "0x3600000000000000000000000000000000000000"

# event MessageSent(bytes) topic0 - carries the message bytes needed for
# attestation + mint.
MESSAGE_SENT_TOPIC0 = "0x2fa9ca894982930190727e75500a97d8dc500233a5065e0f3126c48fbe0343c0"

# Arc's eth_estimateGas can return null for depositForBurn; fall back to a
# generous explicit gas limit instead of failing the bridge.
FALLBACK_GAS_LIMIT = 600_000

DEPOSIT_FOR_BURN_ABI = [{
    "name": "depositForBurn",
    "type": "function",
    "stateMutability": "nonpayable",
    "inputs": [{"name": "amount", "type": "uint256"},
               {"name": "destinationDomain", "type": "uint32"},
               {"name": "mintRecipient", "type": "bytes32"},
               {"name": "burnToken", "type": "address"},
               {"name": "destinationCaller", "type": "bytes32"},
               {"name": "maxFee", "type": "uint256"},
               {"name": "minFinalityThreshold", "type": "uint32"}],
    "outputs": [{"name": "", "type": "uint64"}],
}]

RECEIVE_MESSAGE_ABI = [{
    "name": "receiveMessage",
    "type": "function",
    "stateMutability": "nonpayable",
    "inputs": [{"name": "message", "type": "bytes"},
               {"name": "attestation", "type": "bytes"}],
    "outputs": [{"name": "", "type": "bool"}],
}]


@dataclass
class BridgeResult:
    ok: bool
    tx_id: str = ""
    error: str = ""
    steps: str = ""
    # pending=True: the burn tx is CONFIRMED but the attestation has not
    # landed yet; message_bytes/message_hash let the caller persist the
    # bridge and complete the mint on a later cycle.
    pending: bool = False
    message_hash: str = ""
    message_bytes: str = ""


def burn_token_for(chain: str) -> str | None:
    if chain in USDC_ADDRESSES:
        return USDC_ADDRESSES[chain]
    if chain == "arc":
        return ARC_CCTP_BURN_TOKEN
    return None


def bridge_supported(src_chain: str, dst_chain: str) -> bool:
    both_known = (src_chain in CCTP_DOMAINS and dst_chain in CCTP_DOMAINS
                  and burn_token_for(src_chain) is not None
                  and burn_token_for(dst_chain) is not None)
    # never mix networks: testnet USDC burned on mainnet (or vice versa)
    # would mint nothing
    same_network = (is_testnet_chain(src_chain) == is_testnet_chain(dst_chain))
    return bool(both_known and same_network)


def _cctp_timeout() -> float:
    raw = os.environ.get("MEMICOIN_CCTP_TIMEOUT", "900")
    try:
        return float(raw)
    except ValueError:
        return 900.0


def _min_finality_threshold(src_chain: str) -> int:
    # 1000 = hard finality (Standard). Burns SOURCED FROM Arc require 2000,
    # otherwise the attestation can stay pending forever (docs.arc.io).
    return 2000 if src_chain == "arc" else 1000


def _send_with_fallback_gas(w: EVMWallet, to: str, data: bytes,
                            value_wei: int = 0) -> str:
    try:
        tx = w._base_tx(to, data, value_wei)
    except Exception:
        # Arc (and some RPCs) fail gas estimation for CCTP calls; send with
        # an explicit safe limit rather than aborting.
        addr = w.address()
        tx = {
            "from": addr,
            "to": w.w3.to_checksum_address(to),
            "data": data,
            "value": value_wei,
            "nonce": w.w3.eth.get_transaction_count(addr),
            "chainId": w.cfg["chain_id"],
            "gas": FALLBACK_GAS_LIMIT,
            "gasPrice": w.w3.eth.gas_price,
        }
    return w._send(tx)


def _extract_message_sent(receipt) -> bytes | None:
    """Pull the message bytes from the burn receipt's MessageSent log."""
    for log_entry in receipt.get("logs", []):
        topics = log_entry.get("topics", [])
        if not topics:
            continue
        topic0 = topics[0]
        hexed = topic0.hex() if isinstance(topic0, (bytes, bytearray)) else str(topic0)
        if not hexed.startswith("0x"):
            hexed = "0x" + hexed
        if hexed.lower() != MESSAGE_SENT_TOPIC0:
            continue
        raw = bytes(log_entry["data"])
        # MessageSent(bytes): one dynamic argument -> offset word (0x20),
        # length word, then the payload padded to a 32-byte boundary.
        if len(raw) >= 64:
            msg_len = int.from_bytes(raw[32:64], "big")
            if len(raw) >= 64 + msg_len:
                return raw[64:64 + msg_len]
    return None


def _fetch_attestation(message_hash: str, timeout_s: float,
                       poll_interval_s: float,
                       iris_base: str = IRIS_MAINNET) -> str | None:
    """Poll Circle Iris until the attestation is complete. Returns the
    attestation signature (hex string) or None on timeout."""
    import requests
    url = f"{iris_base}/v2/attestations/{message_hash}"
    deadline = time.time() + timeout_s
    while True:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                body = r.json()
                att = body.get("attestation")
                if body.get("status") == "complete" and att:
                    if isinstance(att, list):
                        att = att[0] if att else None
                    if att:
                        return att
        except Exception as e:
            log.warning("iris attestation fetch failed: %s", e)
        if time.time() + poll_interval_s > deadline:
            return None
        time.sleep(poll_interval_s)


def _mint_on_destination(secrets, dst_chain: str, message_bytes: bytes,
                         attestation: str) -> str:
    """receiveMessage on the destination MessageTransmitterV2 -> mint USDC."""
    w = EVMWallet(secrets, dst_chain)
    if not w.configured():
        raise RuntimeError("EVM wallet key not imported")
    contract = w.w3.eth.contract(
        address=w.w3.to_checksum_address(transmitter_for(dst_chain)),
        abi=RECEIVE_MESSAGE_ABI)
    att = attestation[2:] if attestation.startswith("0x") else attestation
    data = contract.encode_abi("receiveMessage", args=[
        message_bytes, bytes.fromhex(att)])
    return _send_with_fallback_gas(w, MESSAGE_TRANSMITTER_V2, data)


def bridge_usdc(secrets, src_chain: str, dst_chain: str, amount: float,
                attestation_timeout_s: float | None = None,
                poll_interval_s: float = 15.0) -> BridgeResult:
    """Bridge `amount` USDC from src_chain to dst_chain via CCTP V2.

    Burns on the source, polls Iris for the attestation, and mints on the
    destination. If the attestation does not land within the timeout, the
    result is flagged `pending` with the message bytes/hash so the caller
    can persist it and finish the mint on a later cycle.
    """
    if src_chain == dst_chain:
        return BridgeResult(False, error="source and destination are the same chain")
    if not bridge_supported(src_chain, dst_chain):
        return BridgeResult(
            False,
            error=f"CCTP bridging {src_chain} -> {dst_chain} is not supported "
                  f"by this build (needs a verified CCTP domain + USDC burn "
                  f"token on both chains)")
    burn_token = burn_token_for(src_chain)
    w = EVMWallet(secrets, src_chain)
    if not w.configured():
        return BridgeResult(False, error="EVM wallet key not imported")
    try:
        raw_amount = int(round(amount * 10 ** 6))  # USDC: 6 decimals on EVM
        messenger = messenger_for(src_chain)
        w.erc20_approve_if_needed(burn_token, messenger, raw_amount)
        mint_recipient = bytes.fromhex(w.address()[2:].rjust(64, "0"))
        contract = w.w3.eth.contract(
            address=w.w3.to_checksum_address(messenger),
            abi=DEPOSIT_FOR_BURN_ABI)
        data = contract.encode_abi("depositForBurn", args=[
            raw_amount,
            CCTP_DOMAINS[dst_chain],
            mint_recipient,
            w.w3.to_checksum_address(burn_token),
            b"\x00" * 32,  # destinationCaller 0x0: anyone can complete the mint
            0,  # maxFee in burn-token units; 0 = standard (no fast-fee premium)
            _min_finality_threshold(src_chain),
        ])
        burn_tx = _send_with_fallback_gas(w, messenger, data)
        receipt = w.w3.eth.get_transaction_receipt(
            burn_tx if burn_tx.startswith("0x") else "0x" + burn_tx)
        message = _extract_message_sent(receipt)
        if message is None:
            return BridgeResult(
                False, tx_id=burn_tx,
                error=(f"burn tx {burn_tx} confirmed but no MessageSent event "
                       f"found; cannot compute the message hash for the mint. "
                       f"DO NOT retry blindly - check the tx on the explorer."))
        msg_hex = message.hex()
        msg_hash = "0x" + w.w3.keccak(message).hex()
        timeout_s = (_cctp_timeout() if attestation_timeout_s is None
                     else attestation_timeout_s)
        att = _fetch_attestation(msg_hash, timeout_s, poll_interval_s,
                                 iris_base=iris_base_for(src_chain))
        if att is None:
            return BridgeResult(
                False, tx_id=burn_tx, pending=True,
                message_hash=msg_hash, message_bytes=msg_hex,
                error=(f"burn confirmed on {src_chain} ({burn_tx}) but the "
                       f"CCTP attestation did not land within {timeout_s:.0f}s. "
                       f"The mint on {dst_chain} will be completed "
                       f"automatically on a later cycle."))
        mint_tx = _mint_on_destination(secrets, dst_chain, message, att)
        return BridgeResult(
            True, tx_id=burn_tx,
            steps=(f"burned {amount} USDC on {src_chain} ({burn_tx}); minted "
                   f"on {dst_chain} ({mint_tx}) to {w.address()}. Funds are "
                   f"investable on the NEXT decision cycle."))
    except Exception as e:
        log.exception("CCTP bridge failed")
        return BridgeResult(False, error=str(e))


def resume_pending_bridges(db, secrets, per_bridge_timeout_s: float = 30.0,
                           poll_interval_s: float = 5.0) -> list[dict]:
    """Complete pending bridges whose attestation has landed since the burn.

    Called at the start of each on-chain decision cycle. Successfully minted
    bridges are removed from the pending table; ones still waiting (or whose
    mint tx failed) stay pending and are retried next cycle.
    """
    out: list[dict] = []
    for row in db.list_pending_bridges():
        rec = dict(row)
        att = _fetch_attestation(rec["message_hash"], per_bridge_timeout_s,
                                 poll_interval_s)
        if att is None:
            out.append({**rec, "ok": False, "error": "attestation still pending"})
            continue
        try:
            mint_tx = _mint_on_destination(
                secrets, rec["dst_chain"],
                bytes.fromhex(rec["message_bytes"]), att)
            db.remove_pending_bridge(rec["id"])
            out.append({**rec, "ok": True, "mint_tx": mint_tx})
        except Exception as e:
            log.exception("pending bridge mint failed for %s", rec.get("id"))
            out.append({**rec, "ok": False, "error": str(e)})
    return out
