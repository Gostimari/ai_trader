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
        "rpc": "https://ethereum-rpc.publicnode.com",
        "rpc_fallbacks": ["https://eth.drpc.org"],
        "router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2
        "weth": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "explorer": "https://etherscan.io/tx/",
        "native_symbol": "ETH",
    },
    "base": {
        "chain_id": 8453,
        "rpc": "https://mainnet.base.org",
        "rpc_fallbacks": ["https://base-rpc.publicnode.com",
                          "https://base.drpc.org"],
        "router": "0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24",  # Uniswap V2 Router02 on Base
        "weth": "0x4200000000000000000000000000000000000006",
        "explorer": "https://basescan.org/tx/",
        "native_symbol": "ETH",
    },
    "polygon": {
        "chain_id": 137,
        "rpc": "https://polygon-bor-rpc.publicnode.com",
        "rpc_fallbacks": ["https://polygon.drpc.org"],
        "router": "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",  # SushiSwap V2 (UniV2 ABI)
        "weth": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
        "explorer": "https://polygonscan.com/tx/",
        "native_symbol": "POL",
    },
    "binance-smart-chain": {
        "chain_id": 56,
        "rpc": "https://bsc-dataseed.binance.org",
        "rpc_fallbacks": ["https://bsc-rpc.publicnode.com"],
        "router": "0x10ED43C718714eb63d5aA57B78B54704E256024E",  # PancakeSwap V2
        "weth": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
        "explorer": "https://bscscan.com/tx/",
        "native_symbol": "BNB",
    },
    # Arc by Circle (docs.arc.io): EVM-compatible, chain ID 5042, USDC is the
    # native gas token (18 decimals). DEX: ArcSwap, a Uniswap-V2 fork verified
    # live on rpc.mainnet.arc.io (factory() matches its factory; WETH()
    # reverts - ArcSwap pairs token<->USDC directly and the router accepts no
    # native value, so buys route through the canonical USDC contract).
    # allPairsLength() = 0 at verification time: pairs are created on demand,
    # swaps revert until liquidity exists.
    "arc": {
        "chain_id": 5042,
        "rpc": "https://rpc.mainnet.arc.io",
        "router": "0xB433a6F3c690D17E78aa3dD87eC01cdc304278a9",  # ArcSwap (UniV2 fork)
        "weth": None,
        "explorer": "https://explorer.arc.io/tx/",
        "native_symbol": "USDC",
    },
    # Testnet chains (testnet: True): the ONLY safe place to verify real
    # transaction flows before enabling mainnet. CCTP domains mirror the
    # mainnets (sepolia=0, base-sepolia=6) but the V2 contracts are
    # DIFFERENT deployments - see memicoin_core/bridge.py.
    "sepolia": {
        "chain_id": 11155111,
        "rpc": "https://ethereum-sepolia-rpc.publicnode.com",
        "router": None,
        "weth": "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14",
        "explorer": "https://sepolia.etherscan.io/tx/",
        "native_symbol": "ETH",
        "testnet": True,
    },
    "base-sepolia": {
        "chain_id": 84532,
        "rpc": "https://sepolia.base.org",
        "rpc_fallbacks": ["https://base-sepolia-rpc.publicnode.com"],
        "router": None,
        "weth": "0x4200000000000000000000000000000000000006",
        "explorer": "https://sepolia.basescan.org/tx/",
        "native_symbol": "ETH",
        "testnet": True,
    },
}

SOLANA = {
    "rpc": "https://api.mainnet-beta.solana.com",
    "explorer": "https://solscan.io/tx/",
}

SUPPORTED_CHAINS = list(EVM_CHAINS) + ["solana"]
# TON is intentionally NOT in this list: the swap path cannot be safely
# tested from this environment, so we ship an explicit disabled status.


def _checksum(addr: str) -> str:
    """EIP-55 checksum that tolerates ANY input casing. web3's
    to_checksum_address raises on wrong casing; addresses reach us from
    constants, env vars and AI-provided contracts, so normalize first."""
    from web3 import Web3
    return Web3.to_checksum_address(addr.lower())


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
        env = os.environ.get(f"MEMICOIN_RPC_{chain.upper().replace('-', '_')}")
        self._rpcs = ([env] if env
                      else [self.cfg["rpc"]] + list(self.cfg.get("rpc_fallbacks", [])))
        self._rpc_idx = 0
        self._acct = None
        self._connect()

    def _connect(self) -> None:
        from web3 import Web3
        self.w3 = Web3(Web3.HTTPProvider(self._rpcs[self._rpc_idx]))

    def _retry_on_rpc_failure(self, fn):
        """Run an idempotent RPC read; on a connection-level failure rotate
        to the next endpoint for this chain and retry. Public endpoints go
        down often, so one dead endpoint must never freeze the wallet.
        Non-connection errors (reverts, bad addresses) raise immediately."""
        rpcs = getattr(self, "_rpcs", None)
        if not rpcs:  # wallet built without the endpoint list: single shot
            return fn()
        last: Exception | None = None
        for _ in range(len(self._rpcs)):
            try:
                return fn()
            except Exception as e:
                msg = str(e).lower()
                transient = any(k in msg for k in (
                    "connection", "resolve", "resolution", "timeout",
                    "timed out", "max retries", "unreachable", "429",
                    "rate limit"))
                if not transient:
                    raise
                last = e
                self._rpc_idx = (self._rpc_idx + 1) % len(rpcs)
                self._connect()
                log.warning("RPC failed on %s, failing over to %s: %s",
                            self.chain, rpcs[self._rpc_idx], e)
        raise last  # type: ignore[misc]

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
        bal = self._retry_on_rpc_failure(
            lambda: self.w3.eth.get_balance(self.address()))
        return float(self.w3.from_wei(bal, "ether"))

    def erc20_balance(self, token: str) -> float:
        abi = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"}]
        contract = self.w3.eth.contract(
            address=_checksum(token), abi=abi)
        raw = self._retry_on_rpc_failure(
            lambda: contract.functions.balanceOf(self.address()).call())
        decimals = self._erc20_decimals(token)
        return float(raw) / (10 ** decimals)

    def erc20_raw_balance(self, token: str) -> int:
        """Integer balanceOf. The float round-trip through erc20_balance can
        land ABOVE the true balance when converted back to integer (observed
        live: int(37.67070443783452 * 1e18) exceeded the wallet's real BRETT
        balance by 1425 wei, so transferFrom reverted TRANSFER_FROM_FAILED
        even with a MAX allowance). Swaps must never spend more than this."""
        abi = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"}]
        contract = self.w3.eth.contract(
            address=_checksum(token), abi=abi)
        return int(self._retry_on_rpc_failure(
            lambda: contract.functions.balanceOf(self.address()).call()))

    def _erc20_decimals(self, token: str) -> int:
        abi = [{"constant": True, "inputs": [], "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}], "type": "function"}]
        contract = self.w3.eth.contract(
            address=_checksum(token), abi=abi)
        return int(self._retry_on_rpc_failure(
            lambda: contract.functions.decimals().call()))

    # ---------- gas feasibility ----------
    def gas_balance(self) -> float:
        """Native token balance available to pay gas (ETH/POL/BNB/USDC-on-Arc)."""
        return self.native_balance()

    def estimated_gas_cost(self) -> float:
        """Approximate native-token cost of a swap tx on this chain.
        Uses a typical DEX-swap gas limit; conservative (over-estimates)."""
        typical_swap_gas = 250_000  # conservative router swap incl. approval
        try:
            price = self._retry_on_rpc_failure(lambda: self.w3.eth.gas_price)
        except Exception:
            price = 30 * 10 ** 9  # 30 gwei fallback
        return float(self.w3.from_wei(typical_swap_gas * price, "ether"))

    def gas_ok(self, buffer: float = 1.5) -> tuple[bool, str]:
        """Can the wallet pay gas for one swap? Arc passes structurally (USDC
        is the gas token) but its balance still needs to cover the reserve.
        Fails CLOSED: an RPC that cannot be reached means gas cannot be
        verified, so trading is refused rather than attempted blind."""
        native_symbol = self.cfg.get("native_symbol", "ETH")
        try:
            bal = self.gas_balance()
            need = self.estimated_gas_cost() * buffer
        except Exception as e:
            return False, (f"gas check unavailable on {self.chain} "
                           f"(RPC failed: {e}) - refusing to trade blind")
        if native_symbol == "USDC":
            if bal >= need:
                return True, ""
            return False, (f"insufficient USDC on {self.chain} even for gas "
                           f"({bal:.4f} < ~{need:.4f}): top up USDC")
        if bal < need:
            return False, (f"insufficient gas: {bal:.6f} {native_symbol} < "
                           f"~{need:.6f} needed for one swap. Top up {native_symbol} "
                           f"on {self.chain} (a tiny amount suffices) or move "
                           f"activity to Arc where USDC pays gas.")
        return True, ""

    # ---------- gas reserve ----------
    # Adaptive (user design): scale reserved swaps to the balance so small
    # gas wallets keep an investable remainder. Mirrors demowallet's scheme.
    def gas_reserve(self) -> float:
        """Native-token amount held back exclusively for future gas.
        Applies to ALL chains incl. Arc (where USDC itself pays gas)."""
        per_swap = self.estimated_gas_cost()
        balance = self.gas_balance()
        if per_swap <= 0 or balance <= 0:
            return per_swap  # minimum 1 swap
        swaps_left = balance / per_swap
        if swaps_left < 2:
            return per_swap  # min 1 swap
        swaps = max(1, min(3, int(swaps_left / 2)))
        return per_swap * swaps

    def investable_native(self) -> float:
        """How much of the native/gas token the bot may safely spend
        (invest) without stranding the wallet: balance minus the reserve."""
        return max(0.0, self.gas_balance() - self.gas_reserve())

# ---------- writes ----------
    def _send(self, tx: dict) -> str:
        signed = self._account().sign_transaction(tx)
        # eth_account >= 0.13 returns a SignedTransaction dataclass (attribute
        # access); older versions return an AttributeDict (dict access)
        raw = getattr(signed, "raw_transaction", None)
        if raw is None:
            raw = getattr(signed, "rawTransaction", None)
        if raw is None and isinstance(signed, dict):
            raw = signed.get("raw_transaction") or signed.get("rawTransaction")
        if raw is None:
            raise RuntimeError("could not extract raw bytes from signed transaction")
        h = self.w3.eth.send_raw_transaction(raw)
        receipt = self._retry_on_rpc_failure(
            lambda: self.w3.eth.wait_for_transaction_receipt(h, timeout=180))
        if receipt["status"] != 1:
            raise RuntimeError(f"transaction reverted: {h.hex()}")
        return h.hex()

    def _base_tx(self, to: str, data: bytes, value_wei: int = 0) -> dict:
        addr = self.address()
        nonce = self._retry_on_rpc_failure(
            lambda: self.w3.eth.get_transaction_count(addr))
        gas = self._retry_on_rpc_failure(lambda: self.w3.eth.estimate_gas(
            {"from": addr, "to": to, "data": data, "value": value_wei}))
        return {
            "from": addr,
            "to": _checksum(to),
            "data": data,
            "value": value_wei,
            "nonce": nonce,
            "chainId": self.cfg["chain_id"],
            "gas": int(gas * 1.3),
            "gasPrice": self._retry_on_rpc_failure(lambda: self.w3.eth.gas_price),
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
        token_c = self.w3.eth.contract(address=_checksum(token), abi=abi)
        current = self._retry_on_rpc_failure(
            lambda: token_c.functions.allowance(self.address(), router).call())
        if current < amount_raw:
            data = token_c.encode_abi("approve", args=[router, 2 ** 256 - 1])
            self._send(self._base_tx(token, data))
            # the receipt confirms the approve on ONE node - public RPCs are
            # load-balanced and a lagging node can serve the very next swap
            # with the OLD allowance (observed live: a BRETT swap reverted
            # TRANSFER_FROM_FAILED right after its max approve was mined).
            # Re-read until the approval is visible before spending gas.
            import time
            for _ in range(5):
                current = self._retry_on_rpc_failure(
                    lambda: token_c.functions.allowance(
                        self.address(), router).call())
                if current >= amount_raw:
                    return
                time.sleep(1)
            raise RuntimeError(
                f"approve for {token} was mined but the allowance still reads "
                f"below the swap amount - RPC state lag, refusing to send the "
                f"swap into a stale node")

    def _swap_path(self, token_in: str, token_out: str) -> list[str]:
        """Hop list shared by quotes AND swaps - the two must always agree,
        otherwise the quote prices one route while the swap executes another.
        Standard V2 routers route token->token through the wrapped native
        ([in, weth, out]); direct pairs when an endpoint IS the wrapped
        native; ArcSwap has no wrapped native at all - every pair is
        token<->USDC direct."""
        tin = _checksum(token_in)
        tout = _checksum(token_out)
        weth = self._weth()
        if not weth:
            return [tin, tout]
        w = _checksum(weth)
        if tin.lower() == w.lower() or tout.lower() == w.lower():
            return [tin, tout]
        return [tin, w, tout]

    def router_quote_min_out(self, token_in: str, token_out: str,
                             amount_in: float, slippage_pct: float,
                             cap_to_balance: bool = True) -> float:
        """On-chain quote (router getAmountsOut) for a token->token swap,
        lowered by the slippage tolerance. The quoted amount is capped at the
        on-chain balance so quote and swap always agree. cap_to_balance=False
        quotes a NATIVE amount (swapExactETHForTokens sends value with the
        tx; the wallet holds no wrapped-native tokens to cap against).
        Raises when the router has no direct pair for the path; callers fall
        back to market prices. Returns 0.0 when the input is dust."""
        router = self._router()
        if not router:
            raise RuntimeError(self._no_router_error())
        abi = [{"constant": True,
                "inputs": [{"name": "amountIn", "type": "uint256"},
                           {"name": "path", "type": "address[]"}],
                "name": "getAmountsOut",
                "outputs": [{"name": "", "type": "uint256[]"}],
                "type": "function"}]
        dec_in = self._erc20_decimals(token_in)
        path = self._swap_path(token_in, token_out)
        c = self.w3.eth.contract(address=_checksum(router), abi=abi)
        raw_in = int(amount_in * 10 ** dec_in)
        if cap_to_balance:
            raw_in = min(raw_in, self.erc20_raw_balance(token_in))
        if raw_in <= 0:
            return 0.0
        amounts = self._retry_on_rpc_failure(
            lambda: c.functions.getAmountsOut(raw_in, path).call())
        dec_out = self._erc20_decimals(token_out)
        expected = amounts[-1] / 10 ** dec_out
        return expected * (1.0 - slippage_pct / 100.0)

    def swap_exact_tokens_for_tokens(self, token_in: str, token_out: str,
                                     amount_in: float,
                                     min_amount_out: float | None = None) -> TxResult:
        """Uniswap-V2-style swap via the chain router. min_amount_out is the
        caller's slippage bound in OUTPUT-token units; without it the swap
        is refused (amountOutMin=1 would accept any sandwich attack)."""
        try:
            router = self._router()
            if not router:
                return TxResult(False, error=self._no_router_error())
            if min_amount_out is None:
                return TxResult(False, error=(
                    "refusing to swap without a slippage bound: no "
                    "min_amount_out computed from market prices"))
            self.verify_router(router)
            dec_in = self._erc20_decimals(token_in)
            raw_in = min(int(amount_in * 10 ** dec_in),
                         self.erc20_raw_balance(token_in))
            if raw_in <= 0:
                return TxResult(False, error=(
                    f"on-chain balance of {token_in} is zero or dust - "
                    f"nothing to swap"))
            self.erc20_approve_if_needed(token_in, router, raw_in)
            dec_out = self._erc20_decimals(token_out)
            raw_min = max(1, int(min_amount_out * 10 ** dec_out))
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
            path = self._swap_path(token_in, token_out)
            router_c = self.w3.eth.contract(
                address=_checksum(router), abi=abi)
            data = router_c.encode_abi(
                "swapExactTokensForTokens",
                args=[raw_in, raw_min, path, self.address(), deadline])
            tx_id = self._send(self._base_tx(router, data))
            return TxResult(True, tx_id=tx_id, amount=amount_in)
        except Exception as e:
            log.exception("EVM swap failed on %s", self.chain)
            return TxResult(False, error=str(e))

    def verify_router(self, router: str) -> None:
        """Fail-closed safety check: refuse to send funds to an address that
        is not a live Uniswap-V2-style router. A wrong router constant turns
        every swap into a plain transfer to an address nobody controls - the
        funds are unrecoverable. Verified on-chain: the router must hold code
        AND report the configured wrapped-native token via WETH()."""
        code = self._retry_on_rpc_failure(
            lambda: self.w3.eth.get_code(_checksum(router)))
        if not code:
            raise RuntimeError(
                f"router safety check failed on {self.chain}: {router} holds "
                f"NO contract code - refusing to send funds to it")
        weth = self._weth()
        if weth:
            abi = [{"constant": True, "inputs": [], "name": "WETH",
                    "outputs": [{"name": "", "type": "address"}],
                    "type": "function"}]
            c = self.w3.eth.contract(address=_checksum(router),
                                     abi=abi)
            reported = self._retry_on_rpc_failure(lambda: c.functions.WETH().call())
            if isinstance(reported, (bytes, bytearray)):
                reported = "0x" + bytes(reported[-20:]).hex()
            if str(reported).lower() != weth.lower():
                raise RuntimeError(
                    f"router safety check failed on {self.chain}: router "
                    f"{router} reports wrapped-native {reported} but {weth} "
                    f"is configured - refusing to send funds to it")

    def _router(self) -> str | None:
        import os
        if self.cfg.get("router"):
            return self.cfg["router"]
        r = os.environ.get(f"MEMICOIN_{self.chain.upper().replace('-', '_')}_ROUTER")
        return r.lower() if r else r

    def _weth(self) -> str | None:
        import os
        if self.cfg.get("weth"):
            return self.cfg["weth"]
        w = os.environ.get(f"MEMICOIN_{self.chain.upper().replace('-', '_')}_WETH")
        return w.lower() if w else w

    def _no_router_error(self) -> str:
        return (f"No DEX router configured for {self.chain} yet — swaps are disabled "
                f"on this chain until one is set (MEMICOIN_{self.chain.upper().replace('-', '_')}_ROUTER)")

    def swap_exact_eth_for_tokens(self, token_out: str, eth_amount: float,
                                  min_amount_out: float | None = None) -> TxResult:
        """Buy with native currency (ETH/MATIC/BNB/USDC-on-Arc) through the
        wrapped path. Enforces the gas reserve: spending beyond
        investable_native() is refused, never clamped silently. Also refuses
        to trade without a slippage bound (min_amount_out in OUTPUT-token
        units) - amountOutMin=1 would accept any price."""
        try:
            investable = self.investable_native()
            if eth_amount > investable + 1e-12:
                return TxResult(
                    False,
                    error=(f"gas reserve protection: {eth_amount:.6f} exceeds "
                           f"investable {investable:.6f} (reserve "
                           f"{self.gas_reserve():.6f} kept for fees)"))
            router = self._router()
            if not router:
                return TxResult(False, error=self._no_router_error())
            if min_amount_out is None:
                return TxResult(False, error=(
                    "refusing to swap without a slippage bound: no "
                    "min_amount_out computed from market prices"))
            weth = self._weth()
            if not weth:
                return TxResult(False, error=(
                    f"No wrapped-native token configured for {self.chain}: "
                    f"its DEX takes no native value - buys must route "
                    f"through the chain's stablecoin contract instead"))
            self.verify_router(router)
            dec_out = self._erc20_decimals(token_out)
            raw_min = max(1, int(min_amount_out * 10 ** dec_out))
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
            path = [_checksum(weth), _checksum(token_out)]
            router_c = self.w3.eth.contract(
                address=_checksum(router), abi=abi)
            data = router_c.encode_abi("swapExactETHForTokens",
                                       args=[raw_min, path, self.address(), deadline])
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
                self._keypair = Keypair.from_base58_string(raw)
        return self._keypair

    def address(self) -> str:
        return str(self._kp().pubkey())

    def native_balance(self) -> float:
        import os
        from solana.rpc.api import Client
        client = Client(os.environ.get("MEMICOIN_RPC_SOLANA", SOLANA["rpc"]))
        resp = client.get_balance(self._kp().pubkey())
        return float(resp.value) / 1e9

    def _spl_decimals(self, mint: str) -> int:
        import os
        from solana.rpc.api import Client
        from solders.pubkey import Pubkey
        client = Client(os.environ.get("MEMICOIN_RPC_SOLANA", SOLANA["rpc"]))
        resp = client.get_token_supply(Pubkey.from_string(mint))
        return int(resp.value.decimals)

    def swap(self, token_in: str, token_out: str, amount_in: float,
             slippage_bps: int = 100) -> TxResult:
        """Swap via Jupiter aggregator (quote + swap). amount_in is in HUMAN
        token units; it is converted to raw units with the input mint's
        decimals (WSOL = 9 -> lamports)."""
        try:
            import os
            import requests as _req
            from solana.rpc.api import Client
            from solders.keypair import Keypair
            kp = self._kp()
            client = Client(os.environ.get("MEMICOIN_RPC_SOLANA", SOLANA["rpc"]))
            raw_in = int(amount_in * 10 ** self._spl_decimals(token_in))
            params = {"inputMint": token_in, "outputMint": token_out,
                      "amount": raw_in, "slippageBps": max(0, int(slippage_bps))}
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
    # Circle-documented USDC deployments (verified from Circle's docs):
    "ethereum": [("USDC", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6)],
    "base": [("USDC", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", 6)],
    "polygon": [("USDC", "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359", 6)],
    "arc": [("CIRBTC", "0x171a4217b86a807a64eb94757db6849fb4bdbaa0", 18)],
}
# Arc quirk: USDC is the native gas token there (18 decimals) - read as native.
# BSC: no Circle-native USDC; native BNB read via native balance.




# Verified USDC addresses (Circle official deployments only).
USDC_ADDRESSES = {
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "polygon": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    # Circle testnet USDC (faucet.circle.com; 6 decimals like mainnet)
    "sepolia": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",
    "base-sepolia": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
}
# Arc: USDC is the native gas token; the canonical ERC-20 USDC sits at the
# reserved address 0x3600...0000 (verified live: symbol USDC, 6 decimals,
# same ledger as the native balance) - token swaps route through it.
# BSC: no Circle-native USDC deployment; liquidation there falls back to
# wrapped native (WBNB) and is labeled as such in the results.
SOLANA_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZ8nMYp7F"
ARC_USDC = "0x3600000000000000000000000000000000000000"  # canonical USDC on Arc


def stablecoin_target(chain: str) -> tuple[str, str]:
    """(address, label) of the stablecoin to liquidate into on this chain.
    Falls back to the chain's wrapped native token where no verified
    Circle USDC exists; the caller reports the label honestly."""
    if chain == "arc":
        return ARC_USDC, "USDC"
    if chain in USDC_ADDRESSES:
        return USDC_ADDRESSES[chain], "USDC"
    return EVM_CHAINS[chain]["weth"], "wrapped native (no verified USDC on this chain)"


def read_wallet_balances(secrets: SecretStore, chains: list[str] | None = None,
                        extra_contracts: dict | None = None) -> dict:
    """Read native + known token balances for the imported wallet.

    extra_contracts: {chain: [{"address": "0x..", "symbol": "SYM"}, ...]}
    from the user's watchlist — arbitrary tokens the user holds are read
    directly on-chain (balanceOf), so no paid indexer is needed.

    Returns {'evm': {'address': .., 'chains': {chain: {'native': x, 'tokens': [...]}},
             'solana': {...}, 'error': optional}
    Native symbol is per chain (ETH/MATIC/BNB/USDC-on-Arc/SOL).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: dict = {"evm": None, "solana": None}

    def read_evm_chain(chain: str) -> tuple[str, dict]:
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
        # user watchlist contracts: discovered tokens the user wants
        # tracked (e.g. BRETT on Base, ARCAT/ARGUS on Arc). Balance is
        # read directly on-chain; zero balances are simply omitted.
        for extra in (extra_contracts or {}).get(chain, []):
            if any(t["contract"].lower() == extra["address"].lower()
                   for t in entry["tokens"]):
                continue
            try:
                bal = w.erc20_balance(extra["address"])
                if bal and bal > 0:
                    entry["tokens"].append({"symbol": extra.get("symbol", "?"),
                                            "contract": extra["address"],
                                            "balance": bal})
            except Exception:
                continue
        return chain, entry

    def read_solana() -> dict:
        sw = SolanaWallet(secrets)
        sol = {"address": sw.address(), "native": 0.0, "tokens": []}
        try:
            sol["native"] = sw.native_balance()
        except Exception:
            pass
        return sol

    jobs = []
    if secrets.has("onchain_private_key"):
        target = [c for c in (chains or list(EVM_CHAINS)) if c in EVM_CHAINS]
        jobs += [("evm", c, read_evm_chain, (c,)) for c in target]
    if secrets.has("onchain_solana_private_key"):
        jobs.append(("sol", None, read_solana, ()))

    if not jobs:
        return out
    evm = None
    if secrets.has("onchain_private_key"):
        evm = {"address": None, "chains": {}}
        try:
            from web3 import Web3
            evm["address"] = Web3().eth.account.from_key(
                secrets.get("onchain_private_key")).address
        except Exception as e:
            out["error"] = str(e)
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        futures = {pool.submit(fn, *args): (kind, key)
                   for kind, key, fn, args in jobs}
        for fut in as_completed(futures):
            kind, key = futures[fut]
            try:
                if kind == "evm":
                    chain, entry = fut.result()
                    evm["chains"][chain] = entry
                else:
                    out["solana"] = fut.result()
            except Exception as e:
                out["error"] = str(e)
    out["evm"] = evm
    return out


def build_extra_contracts(secrets: SecretStore, market=None) -> dict[str, list[dict]]:
    """Everything the wallet read should track beyond the built-in token list:
    the user's watchlist contracts (from the market asset map) PLUS automatic
    discovery of every ERC-20 the imported wallet ever transacted (free
    Etherscan V2 key). Returns {chain: [{"address": .., "symbol": ..}]}.
    Actual balances are always read on-chain afterwards; discovery only
    supplies the candidate contracts."""
    extra: dict[str, list[dict]] = {}
    try:
        for a in (market.get_asset_map() or {}).values() if market else []:
            contract = a.get("contract") or ""
            chain = a.get("chain") or ""
            if contract.startswith("0x") and len(contract) == 42 and chain:
                bucket = extra.setdefault(chain, [])
                if not any(t["address"].lower() == contract.lower() for t in bucket):
                    bucket.append({"address": contract, "symbol": a["asset"]})
    except Exception:
        pass
    try:
        if secrets.has("onchain_private_key") and secrets.has("etherscan_api_key"):
            from web3 import Web3
            addr = Web3().eth.account.from_key(
                secrets.get("onchain_private_key")).address
            from .discovery import discover_evm_tokens
            discovered = discover_evm_tokens(
                addr, secrets.get("etherscan_api_key")) or {}
            for chain, toks in discovered.items():
                bucket = extra.setdefault(chain, [])
                known = {t["address"].lower() for t in bucket}
                for t in toks:
                    if t["address"].lower() not in known:
                        bucket.append(t)
    except Exception:
        pass
    return extra
