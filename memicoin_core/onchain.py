"""On-chain wallet module (§4.3 Mode B): non-custodial, imported hot wallet.

The private key is imported once, stored ONLY in the encrypted SecretStore,
and used exclusively for signing transactions locally. It is never logged,
never sent anywhere, and never returned by any API.

EVM chains (Ethereum/Base/Polygon/BSC) share one web3 stack; Solana uses
solders keypairs. Router addresses and RPC endpoints are per-chain config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .secrets import SecretStore

log = logging.getLogger("memicoin.onchain")

# ---------------------------------------------------------------------------
# Chain registry: RPC endpoints (public), routers, wrapped native tokens.
# Users can override RPCs via env vars: MEMICOIN_RPC_<CHAIN>
# ---------------------------------------------------------------------------
EVM_CHAINS = {
    "ethereum": {
        "chain_id": 1,
        "rpc": "https://eth.llamarpc.com",
        "router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2
        "weth": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "explorer": "https://etherscan.io/tx/",
    },
    "base": {
        "chain_id": 8453,
        "rpc": "https://mainnet.base.org",
        "router": "0x4752ba5DBc36f75D9e232F321687625204392e48",  # Uniswap V2 on Base
        "weth": "0x4200000000000000000000000000000000000006",
        "explorer": "https://basescan.org/tx/",
    },
    "polygon": {
        "chain_id": 137,
        "rpc": "https://polygon-rpc.com",
        "router": "0xa5E0829CaCEd8fAdd424714229311691BfBe27c8",  # QuickSwap (UniV2)
        "weth": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
        "explorer": "https://polygonscan.com/tx/",
    },
    "binance-smart-chain": {
        "chain_id": 56,
        "rpc": "https://bsc-dataseed.binance.org",
        "router": "0x10ED43C718714eb63d5aA57B78B54704E256024E",  # PancakeSwap V2
        "weth": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
        "explorer": "https://bscscan.com/tx/",
    },
    # Arc by Circle (docs.arc.io): EVM-compatible, chain ID 5042, USDC is the
    # native gas token (18 decimals). No meme-coin DEX router exists there yet
    # (the chain is stablecoin-native; Circle's Swap Kit is stablecoin-only),
    # so router/weth stay unset: wallet import and balance reads work, swaps
    # are refused with a clear error until a router is configured via
    # MEMICOIN_ARC_ROUTER / MEMICOIN_ARC_WETH.
    "arc": {
        "chain_id": 5042,
        "rpc": "https://rpc.mainnet.arc.io",
        "router": None,
        "weth": None,
        "explorer": "https://explorer.arc.io/tx/",
        "native_symbol": "USDC",
    },
}

SOLANA = {
    "rpc": "https://api.mainnet-beta.solana.com",
    "explorer": "https://solscan.io/tx/",
}

SUPPORTED_CHAINS = list(EVM_CHAINS) + ["solana"]
# TON is intentionally NOT in this list: the swap path cannot be safely
# tested from this environment, so we ship an explicit disabled status.


@dataclass
class TxResult:
    ok: bool
    tx_id: str = ""
    amount: float = 0.0
    value_usd: float = 0.0
    error: str = ""


class EVMWallet:
    def __init__(self, secrets: SecretStore, chain: str):
        assert chain in EVM_CHAINS, f"unsupported EVM chain {chain}"
        self.secrets = secrets
        self.chain = chain
        self.cfg = EVM_CHAINS[chain]
        import os
        rpc = os.environ.get(f"MEMICOIN_RPC_{chain.upper().replace('-', '_')}", self.cfg["rpc"])
        from web3 import Web3
        self.w3 = Web3(Web3.HTTPProvider(rpc))
        self._acct = None

    def _account(self):
        if self._acct is None:
            key = self.secrets.get("onchain_private_key")
            if not key:
                raise RuntimeError("on-chain private key not imported")
            if not key.startswith("0x"):
                key = "0x" + key
            self._acct = self.w3.eth.account.from_key(key)
        return self._acct

    def address(self) -> str:
        return self._account().address

    def configured(self) -> bool:
        return self.secrets.has("onchain_private_key")

    # ---------- reads ----------
    def native_balance(self) -> float:
        bal = self.w3.eth.get_balance(self.address())
        return float(self.w3.from_wei(bal, "ether"))

    def erc20_balance(self, token: str) -> float:
        abi = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"}]
        contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token), abi=abi)
        raw = contract.functions.balanceOf(self.address()).call()
        decimals = self._erc20_decimals(token)
        return float(raw) / (10 ** decimals)

    def _erc20_decimals(self, token: str) -> int:
        abi = [{"constant": True, "inputs": [], "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}], "type": "function"}]
        contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token), abi=abi)
        return int(contract.functions.decimals().call())

    # ---------- writes ----------
    def _send(self, tx: dict) -> str:
        signed = self._account().sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed["rawTransaction"])
        receipt = self.w3.eth.wait_for_transaction_receipt(h, timeout=180)
        if receipt["status"] != 1:
            raise RuntimeError(f"transaction reverted: {h.hex()}")
        return h.hex()

    def _base_tx(self, to: str, data: bytes, value_wei: int = 0) -> dict:
        addr = self.address()
        nonce = self.w3.eth.get_transaction_count(addr)
        gas = self.w3.eth.estimate_gas({"from": addr, "to": to, "data": data, "value": value_wei})
        return {
            "from": addr,
            "to": self.w3.to_checksum_address(to),
            "data": data,
            "value": value_wei,
            "nonce": nonce,
            "chainId": self.cfg["chain_id"],
            "gas": int(gas * 1.3),
            "gasPrice": self.w3.eth.gas_price,
        }

    def erc20_approve_if_needed(self, token: str, router: str, amount_raw: int) -> None:
        abi = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"},
                                             {"name": "_spender", "type": "address"}],
                "name": "allowance", "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"},
               {"constant": False,
                "inputs": [{"name": "_spender", "type": "address"},
                           {"name": "_value", "type": "uint256"}],
                "name": "approve", "outputs": [{"name": "", "type": "bool"}],
                "type": "function"}]
        token_c = self.w3.eth.contract(address=self.w3.to_checksum_address(token), abi=abi)
        current = token_c.functions.allowance(self.address(), router).call()
        if current < amount_raw:
            data = token_c.encode_abi("approve", args=[router, 2 ** 256 - 1])
            self._send(self._base_tx(token, data))

    def swap_exact_tokens_for_tokens(self, token_in: str, token_out: str,
                                     amount_in: float) -> TxResult:
        """Uniswap-V2-style swap via the chain router."""
        try:
            router = self._router()
            if not router:
                return TxResult(False, error=self._no_router_error())
            dec_in = self._erc20_decimals(token_in)
            raw_in = int(amount_in * 10 ** dec_in)
            self.erc20_approve_if_needed(token_in, router, raw_in)
            dec_out = self._erc20_decimals(token_out)
            abi = [{"constant": False,
                    "inputs": [{"name": "amountIn", "type": "uint256"},
                               {"name": "amountOutMin", "type": "uint256"},
                               {"name": "path", "type": "address[]"},
                               {"name": "to", "type": "address"},
                               {"name": "deadline", "type": "uint256"}],
                    "name": "swapExactTokensForTokens",
                    "outputs": [{"name": "", "type": "uint256[]"}],
                    "type": "function"}]
            import time
            deadline = int(time.time()) + 1200
            path = [self.w3.to_checksum_address(token_in),
                    self.cfg["weth"],
                    self.w3.to_checksum_address(token_out)]
            router_c = self.w3.eth.contract(
                address=self.w3.to_checksum_address(router), abi=abi)
            data = router_c.encode_abi(
                "swapExactTokensForTokens",
                args=[raw_in, 1, path, self.address(), deadline])
            tx_id = self._send(self._base_tx(router, data))
            return TxResult(True, tx_id=tx_id, amount=amount_in)
        except Exception as e:
            log.exception("EVM swap failed on %s", self.chain)
            return TxResult(False, error=str(e))

    def _router(self) -> str | None:
        import os
        if self.cfg.get("router"):
            return self.cfg["router"]
        return os.environ.get(f"MEMICOIN_{self.chain.upper().replace('-', '_')}_ROUTER")

    def _weth(self) -> str | None:
        import os
        if self.cfg.get("weth"):
            return self.cfg["weth"]
        return os.environ.get(f"MEMICOIN_{self.chain.upper().replace('-', '_')}_WETH")

    def _no_router_error(self) -> str:
        return (f"No DEX router configured for {self.chain} yet — swaps are disabled "
                f"on this chain until one is set (MEMICOIN_{self.chain.upper().replace('-', '_')}_ROUTER)")

    def swap_exact_eth_for_tokens(self, token_out: str, eth_amount: float) -> TxResult:
        """Buy with native currency (ETH/MATIC/BNB) through WETH path."""
        try:
            router = self._router()
            if not router:
                return TxResult(False, error=self._no_router_error())
            weth = self._weth()
            if not weth:
                return TxResult(False, error=f"No wrapped-native token configured for {self.chain}")
            abi = [{"constant": False,
                    "inputs": [{"name": "amountOutMin", "type": "uint256"},
                               {"name": "path", "type": "address[]"},
                               {"name": "to", "type": "address"},
                               {"name": "deadline", "type": "uint256"}],
                    "name": "swapExactETHForTokens",
                    "outputs": [{"name": "", "type": "uint256[]"}],
                    "type": "function"}]
            import time
            deadline = int(time.time()) + 1200
            path = [self.w3.to_checksum_address(weth), self.w3.to_checksum_address(token_out)]
            router_c = self.w3.eth.contract(
                address=self.w3.to_checksum_address(router), abi=abi)
            data = router_c.encode_abi("swapExactETHForTokens",
                                       args=[1, path, self.address(), deadline])
            value_wei = self.w3.to_wei(eth_amount, "ether")
            tx_id = self._send(self._base_tx(router, data, value_wei))
            return TxResult(True, tx_id=tx_id, amount=eth_amount)
        except Exception as e:
            log.exception("EVM native swap failed on %s", self.chain)
            return TxResult(False, error=str(e))


class SolanaWallet:
    def __init__(self, secrets: SecretStore):
        self.secrets = secrets
        self._keypair = None

    def configured(self) -> bool:
        return self.secrets.has("onchain_solana_private_key")

    def _kp(self):
        if self._keypair is None:
            raw = self.secrets.get("onchain_solana_private_key")
            if not raw:
                raise RuntimeError("solana private key not imported")
            import json as _json
            from solders.keypair import Keypair
            try:
                nums = _json.loads(raw)
                self._keypair = Keypair.from_bytes(bytes(nums))
            except Exception:
                self._keypair = Keypair.from_base58_text(raw)
        return self._keypair

    def address(self) -> str:
        return str(self._kp().pubkey())

    def native_balance(self) -> float:
        import os
        from solana.rpc.api import Client
        client = Client(os.environ.get("MEMICOIN_RPC_SOLANA", SOLANA["rpc"]))
        resp = client.get_balance(self._kp().pubkey())
        return float(resp.value) / 1e9

    def swap(self, token_in: str, token_out: str, amount_in: float) -> TxResult:
        """Swap via Jupiter aggregator (quote + swap)."""
        try:
            import os
            import requests as _req
            from solana.rpc.api import Client
            from solders.keypair import Keypair
            kp = self._kp()
            client = Client(os.environ.get("MEMICOIN_RPC_SOLANA", SOLANA["rpc"]))
            params = {"inputMint": token_in, "outputMint": token_out,
                      "amount": int(amount_in), "slippageBps": 100}
            quote = _req.get("https://quote-api.jup.ag/v6/quote", params=params, timeout=20)
            quote.raise_for_status()
            swap_body = {"quoteResponse": quote.json(),
                         "userPublicKey": str(kp.pubkey()),
                         "wrapAndUnwrapSol": True}
            swap = _req.post("https://quote-api.jup.ag/v6/swap",
                             json=swap_body, timeout=30)
            swap.raise_for_status()
            tx_b64 = swap.json()["swapTransaction"]
            import base64 as _b64
            from solana.transaction import Transaction
            from solders.message import to_bytes
            tx = Transaction.deserialize(_b64.b64decode(tx_b64))
            signed = tx.sign(kp)
            resp = client.send_raw_transaction(signed)
            return TxResult(True, tx_id=str(resp.value))
        except Exception as e:
            log.exception("Solana swap failed")
            return TxResult(False, error=str(e))


def wallet_for(chain: str, secrets: SecretStore):
    if chain in EVM_CHAINS:
        return EVMWallet(secrets, chain)
    if chain == "solana":
        return SolanaWallet(secrets)
    if chain == "ton":
        raise RuntimeError("TON support is not enabled yet (untestable swap path)")
    raise RuntimeError(f"unsupported chain: {chain}")


# ---------------------------------------------------------------------------
# Wallet balance reading (real-money mode): native + token balances per chain
# ---------------------------------------------------------------------------
# Token lists are intentionally minimal and hardcoded to well-known assets.
# Discovering arbitrary tokens requires an indexer API (per-chain, key-based);
# the watchlist covers everything else the user explicitly tracks.
# Only contracts verified from official docs are listed. Unknown/unverified
# addresses must never be hardcoded here - the watchlist (user-supplied,
# DexScreener-indexed) covers additional tokens.
EVM_TOKENS = {
    "arc": [("CIRBTC", "0x171a4217b86a807a64eb94757db6849fb4bdbaa0", 18)],
}
# Arc quirk: USDC is the native gas token there (18 decimals) - read as native.




# Verified USDC addresses (Circle official deployments only).
USDC_ADDRESSES = {
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "polygon": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
}
# Arc: USDC is the native gas token — no contract needed.
# BSC: no Circle-native USDC deployment; liquidation there falls back to
# wrapped native (WBNB) and is labeled as such in the results.
SOLANA_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZ8nMYp7F"


def stablecoin_target(chain: str) -> tuple[str, str]:
    """(address, label) of the stablecoin to liquidate into on this chain.
    Falls back to the chain's wrapped native token where no verified
    Circle USDC exists; the caller reports the label honestly."""
    if chain == "arc":
        return EVM_CHAINS["arc"]["native_symbol"], "native USDC"
    if chain in USDC_ADDRESSES:
        return USDC_ADDRESSES[chain], "USDC"
    return EVM_CHAINS[chain]["weth"], "wrapped native (no verified USDC on this chain)"


def read_wallet_balances(secrets: SecretStore, chains: list[str] | None = None) -> dict:
    """Read native + known token balances for the imported wallet.

    Returns {'evm': {'address': .., 'chains': {chain: {'native': x, 'tokens': [... KW..]}}},
             'solana': {...}, 'error': optional}
    Native symbol is per chain (ETH/MATIC/BNB/USDC-on-Arc/SOL).
    """
    out: dict = {"evm": None, "solana": None}
    if secrets.has("onchain_private_key"):
        evm = {"address": None, "chains": {}}
        try:
            from web3 import Web3
            acct = Web3().eth.account.from_key(secrets.get("onchain_private_key"))
            evm["address"] = acct.address
            target = chains or list(EVM_CHAINS)
            for chain in target:
                if chain not in EVM_CHAINS:
                    continue
                w = EVMWallet(secrets, chain)
                native_sym = EVM_CHAINS[chain].get("native_symbol", "ETH")
                entry = {"native_symbol": native_sym, "native": 0.0, "tokens": []}
                try:
                    entry["native"] = w.native_balance()
                except Exception:
                    pass
                for sym, addr, decimals in EVM_TOKENS.get(chain, []):
                    try:
                        bal = w.erc20_balance(addr)
                        if bal and bal > 0:
                            entry["tokens"].append({"symbol": sym, "contract": addr,
                                                    "balance": bal})
                    except Exception:
                        continue
                evm["chains"][chain] = entry
            out["evm"] = evm
        except Exception as e:
            out["error"] = str(e)
    if secrets.has("onchain_solana_private_key"):
        try:
            sw = SolanaWallet(secrets)
            sol = {"address": sw.address(), "native": 0.0, "tokens": []}
            try:
                sol["native"] = sw.native_balance()
            except Exception:
                pass
            out["solana"] = sol
        except Exception as e:
            out["error"] = str(e)
    return out
