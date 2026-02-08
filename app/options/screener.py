"""
Options Screener Module
Scans the market for high-potential options trades within a given budget.
Uses Yahoo Finance for stock and options chain data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


# Default watchlist: high-volume, options-active tickers popular on Robinhood
DEFAULT_WATCHLIST = [
    # Mega-cap tech (high liquidity, tight spreads)
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    # High-beta / momentum names
    "COIN", "MARA", "RIOT", "PLTR", "SOFI", "SNAP", "PINS", "ROKU", "SQ", "SHOP",
    "HOOD", "DKNG", "RBLX", "U", "CRWD", "SNOW", "NET", "ENPH", "FSLR", "RIVN",
    "LCID", "NIO", "XPEV", "LI", "BABA", "JD", "PDD",
    # Biotech / pharma (catalyst-driven)
    "MRNA", "BNTX", "BIIB",
    # ETFs (broad market plays)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "GLD", "SLV", "TLT",
    "ARKK", "SOXL", "TQQQ",
    # Meme / retail favorites
    "GME", "AMC", "BBBY", "WISH", "CLOV",
]


class OptionsScreener:
    """Screens the market for options trading opportunities within budget."""

    def __init__(self, budget: float = 500.0, watchlist: Optional[List[str]] = None):
        self.budget = budget
        self.max_premium = budget / 100  # per-share premium (1 contract = 100 shares)
        self.watchlist = watchlist or DEFAULT_WATCHLIST

    def scan_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Analyze a single ticker for options potential.
        Returns a summary dict or None if the ticker doesn't qualify.
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")

            if hist.empty or len(hist) < 5:
                return None

            info = stock.info or {}
            current_price = hist["Close"].iloc[-1]

            # Calculate beta (vs SPY proxy using rolling returns)
            beta = info.get("beta", None)

            # Calculate historical volatility (20-day)
            returns = hist["Close"].pct_change().dropna()
            hist_vol_20d = returns.std() * np.sqrt(252) if len(returns) >= 5 else 0

            # Average daily volume
            avg_volume = hist["Volume"].mean()

            # Price momentum (5-day)
            price_5d_ago = hist["Close"].iloc[-6] if len(hist) >= 6 else hist["Close"].iloc[0]
            momentum_5d = (current_price - price_5d_ago) / price_5d_ago

            # Price momentum (1-day)
            price_1d_ago = hist["Close"].iloc[-2] if len(hist) >= 2 else current_price
            momentum_1d = (current_price - price_1d_ago) / price_1d_ago

            # ATR (Average True Range) for expected daily move
            high = hist["High"]
            low = hist["Low"]
            close = hist["Close"]
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else tr.mean()
            atr_pct = atr_14 / current_price

            # Get options expiration dates
            try:
                expirations = stock.options
            except Exception:
                expirations = []

            if not expirations:
                return None

            # Find nearest weekly/short-term expiration (0-7 days for day trades)
            today = datetime.now().date()
            near_term_exps = []
            swing_exps = []
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                days_to_exp = (exp_date - today).days
                if 0 <= days_to_exp <= 7:
                    near_term_exps.append(exp_str)
                elif 7 < days_to_exp <= 30:
                    swing_exps.append(exp_str)

            # Get implied volatility from nearest options chain
            iv_data = self._get_iv_data(stock, near_term_exps or swing_exps, current_price)

            return {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "beta": round(beta, 2) if beta else None,
                "hist_volatility_20d": round(hist_vol_20d, 4),
                "implied_volatility": iv_data.get("avg_iv"),
                "iv_rank": iv_data.get("iv_rank"),
                "avg_volume": int(avg_volume),
                "momentum_1d": round(momentum_1d, 4),
                "momentum_5d": round(momentum_5d, 4),
                "atr_14": round(atr_14, 2),
                "atr_pct": round(atr_pct, 4),
                "near_term_expirations": near_term_exps,
                "swing_expirations": swing_exps,
                "has_affordable_options": iv_data.get("has_affordable", False),
                "cheapest_call_premium": iv_data.get("cheapest_call"),
                "cheapest_put_premium": iv_data.get("cheapest_put"),
            }

        except Exception as e:
            logger.warning(f"Error scanning {ticker}: {e}")
            return None

    def _get_iv_data(self, stock, expirations: List[str], current_price: float) -> Dict:
        """Extract implied volatility data from the options chain."""
        if not expirations:
            return {"avg_iv": None, "iv_rank": None, "has_affordable": False}

        try:
            exp = expirations[0]
            chain = stock.option_chain(exp)
            calls = chain.calls
            puts = chain.puts

            # Filter to near-the-money options (within 10% of current price)
            ntm_calls = calls[
                (calls["strike"] >= current_price * 0.90) &
                (calls["strike"] <= current_price * 1.10)
            ]
            ntm_puts = puts[
                (puts["strike"] >= current_price * 0.90) &
                (puts["strike"] <= current_price * 1.10)
            ]

            # Average IV from near-the-money options
            all_iv = pd.concat([
                ntm_calls["impliedVolatility"] if "impliedVolatility" in ntm_calls.columns else pd.Series(dtype=float),
                ntm_puts["impliedVolatility"] if "impliedVolatility" in ntm_puts.columns else pd.Series(dtype=float),
            ])
            avg_iv = round(all_iv.mean(), 4) if not all_iv.empty else None

            # Check for affordable options within budget
            affordable_calls = calls[calls["lastPrice"] <= self.max_premium]
            affordable_puts = puts[puts["lastPrice"] <= self.max_premium]
            has_affordable = len(affordable_calls) > 0 or len(affordable_puts) > 0

            # Cheapest near-the-money options
            cheapest_call = None
            cheapest_put = None
            if not ntm_calls.empty:
                affordable_ntm_calls = ntm_calls[ntm_calls["lastPrice"] <= self.max_premium]
                if not affordable_ntm_calls.empty:
                    cheapest_call = round(affordable_ntm_calls["lastPrice"].min(), 2)
            if not ntm_puts.empty:
                affordable_ntm_puts = ntm_puts[ntm_puts["lastPrice"] <= self.max_premium]
                if not affordable_ntm_puts.empty:
                    cheapest_put = round(affordable_ntm_puts["lastPrice"].min(), 2)

            # Simple IV rank estimate (compare current IV to 20-day hist vol)
            iv_rank = None
            if avg_iv:
                hist = stock.history(period="3mo")
                if not hist.empty:
                    returns = hist["Close"].pct_change().dropna()
                    hv_90d = returns.std() * np.sqrt(252)
                    if hv_90d > 0:
                        iv_rank = round(avg_iv / hv_90d, 2)

            return {
                "avg_iv": avg_iv,
                "iv_rank": iv_rank,
                "has_affordable": has_affordable,
                "cheapest_call": cheapest_call,
                "cheapest_put": cheapest_put,
            }

        except Exception as e:
            logger.warning(f"Error getting IV data: {e}")
            return {"avg_iv": None, "iv_rank": None, "has_affordable": False}

    def run_scan(self, max_workers: int = 8) -> List[Dict]:
        """
        Run the full screener across the watchlist.
        Returns a list of qualifying tickers sorted by a composite score.
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.scan_ticker, ticker): ticker
                for ticker in self.watchlist
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result and result["has_affordable_options"]:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to scan {ticker}: {e}")

        # Score and rank
        for r in results:
            r["composite_score"] = self._calculate_composite_score(r)

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results

    def _calculate_composite_score(self, data: Dict) -> float:
        """
        Calculate a composite opportunity score (0-100) based on:
        - Volatility (higher = more opportunity for day trades)
        - Volume (higher = better fills on Robinhood)
        - Momentum magnitude (bigger moves = bigger potential)
        - IV rank (elevated IV = higher premiums, potential mean reversion)
        - ATR% (daily move potential)
        """
        score = 0.0

        # Volatility component (0-25 points)
        hv = data.get("hist_volatility_20d", 0)
        score += min(hv * 50, 25)

        # Volume component (0-20 points)
        vol = data.get("avg_volume", 0)
        if vol > 50_000_000:
            score += 20
        elif vol > 10_000_000:
            score += 15
        elif vol > 5_000_000:
            score += 10
        elif vol > 1_000_000:
            score += 5

        # Momentum magnitude (0-20 points) - absolute value, both directions are opportunities
        mom_1d = abs(data.get("momentum_1d", 0))
        mom_5d = abs(data.get("momentum_5d", 0))
        score += min(mom_1d * 200, 10)
        score += min(mom_5d * 40, 10)

        # IV rank component (0-15 points)
        iv_rank = data.get("iv_rank")
        if iv_rank and iv_rank > 1.5:
            score += 15  # IV much higher than HV - big move expected
        elif iv_rank and iv_rank > 1.2:
            score += 10
        elif iv_rank and iv_rank > 1.0:
            score += 5

        # ATR% component (0-20 points) - daily move potential
        atr_pct = data.get("atr_pct", 0)
        score += min(atr_pct * 400, 20)

        return round(score, 2)

    def get_top_candidates(self, n: int = 10) -> List[Dict]:
        """Get the top N candidates from the scan."""
        all_results = self.run_scan()
        return all_results[:n]
