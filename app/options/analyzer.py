"""
Options Analyzer Module
Deep-dives into specific tickers to analyze options chains, Greeks,
probability of profit, and recommends optimal strategies for a given budget.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """Analyzes options chains and recommends strategies for a given ticker."""

    def __init__(self, budget: float = 500.0, risk_free_rate: float = 0.05):
        self.budget = budget
        self.max_premium = budget / 100
        self.risk_free_rate = risk_free_rate

    def analyze_ticker(self, ticker: str, expiration: Optional[str] = None) -> Dict:
        """
        Full options analysis for a ticker.
        Returns Greeks, probability metrics, and strategy recommendations.
        """
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")

        if hist.empty:
            raise ValueError(f"No price data for {ticker}")

        current_price = hist["Close"].iloc[-1]
        returns = hist["Close"].pct_change().dropna()
        hist_vol = returns.std() * np.sqrt(252)

        # Get expiration
        expirations = stock.options
        if not expirations:
            raise ValueError(f"No options available for {ticker}")

        if expiration and expiration in expirations:
            target_exp = expiration
        else:
            # Pick the nearest expiration within 7 days, or the closest one
            today = datetime.now().date()
            best_exp = None
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                days_out = (exp_date - today).days
                if 0 <= days_out <= 7:
                    best_exp = exp_str
                    break
            if not best_exp:
                best_exp = expirations[0]
            target_exp = best_exp

        exp_date = datetime.strptime(target_exp, "%Y-%m-%d").date()
        days_to_exp = max((exp_date - datetime.now().date()).days, 1)
        years_to_exp = days_to_exp / 365.0

        # Get options chain
        chain = stock.option_chain(target_exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        # Analyze calls and puts
        analyzed_calls = self._analyze_chain(
            calls, current_price, hist_vol, years_to_exp, "call"
        )
        analyzed_puts = self._analyze_chain(
            puts, current_price, hist_vol, years_to_exp, "put"
        )

        # Filter to affordable options
        affordable_calls = [c for c in analyzed_calls if c["total_cost"] <= self.budget]
        affordable_puts = [p for p in analyzed_puts if p["total_cost"] <= self.budget]

        # Generate strategy recommendations
        strategies = self._recommend_strategies(
            ticker, current_price, hist_vol, hist, days_to_exp,
            affordable_calls, affordable_puts, calls, puts, target_exp
        )

        # Technical bias
        bias = self._get_technical_bias(hist)

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "historical_volatility": round(hist_vol, 4),
            "expiration": target_exp,
            "days_to_expiration": days_to_exp,
            "technical_bias": bias,
            "top_calls": sorted(affordable_calls, key=lambda x: x["score"], reverse=True)[:5],
            "top_puts": sorted(affordable_puts, key=lambda x: x["score"], reverse=True)[:5],
            "strategies": strategies,
        }

    def _analyze_chain(
        self,
        chain: pd.DataFrame,
        spot: float,
        vol: float,
        T: float,
        option_type: str,
    ) -> List[Dict]:
        """Analyze each contract in an options chain."""
        results = []

        for _, row in chain.iterrows():
            strike = row["strike"]
            premium = row.get("lastPrice", 0)
            bid = row.get("bid", 0)
            ask = row.get("ask", 0)
            volume = row.get("volume", 0)
            open_interest = row.get("openInterest", 0)
            iv = row.get("impliedVolatility", vol)

            if premium <= 0 or np.isnan(premium):
                continue

            total_cost = premium * 100

            # Calculate Greeks using Black-Scholes
            greeks = self._calculate_greeks(spot, strike, T, self.risk_free_rate, iv, option_type)

            # Probability of profit
            pop = self._probability_of_profit(spot, strike, premium, T, iv, option_type)

            # Expected move to breakeven
            if option_type == "call":
                breakeven = strike + premium
                pct_to_breakeven = (breakeven - spot) / spot
            else:
                breakeven = strike - premium
                pct_to_breakeven = (spot - breakeven) / spot

            # Risk/reward ratio
            max_loss = total_cost
            # Estimate max gain as 2x premium for a day trade (conservative)
            estimated_gain_2x = total_cost
            risk_reward = estimated_gain_2x / max_loss if max_loss > 0 else 0

            # Composite score
            score = self._score_contract(
                greeks, pop, volume or 0, open_interest or 0,
                bid, ask, premium, pct_to_breakeven
            )

            results.append({
                "strike": strike,
                "premium": round(premium, 2),
                "total_cost": round(total_cost, 2),
                "bid": round(bid, 2) if bid else 0,
                "ask": round(ask, 2) if ask else 0,
                "spread": round(ask - bid, 2) if (ask and bid) else None,
                "volume": int(volume) if volume and not np.isnan(volume) else 0,
                "open_interest": int(open_interest) if open_interest and not np.isnan(open_interest) else 0,
                "implied_volatility": round(iv, 4) if iv else None,
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "theta": greeks["theta"],
                "vega": greeks["vega"],
                "probability_of_profit": round(pop, 4),
                "breakeven": round(breakeven, 2),
                "pct_to_breakeven": round(pct_to_breakeven, 4),
                "max_loss": round(max_loss, 2),
                "score": round(score, 2),
                "type": option_type,
            })

        return results

    def _calculate_greeks(
        self, S: float, K: float, T: float, r: float, sigma: float, option_type: str
    ) -> Dict:
        """Calculate Black-Scholes Greeks."""
        if sigma <= 0 or T <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (
                (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = (
                (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
        }

    def _probability_of_profit(
        self, spot: float, strike: float, premium: float,
        T: float, sigma: float, option_type: str
    ) -> float:
        """
        Estimate probability of profit at expiration using log-normal distribution.
        """
        if sigma <= 0 or T <= 0:
            return 0.0

        if option_type == "call":
            breakeven = strike + premium
            # P(S_T > breakeven)
            d = (np.log(spot / breakeven) + (self.risk_free_rate - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return norm.cdf(d)
        else:
            breakeven = strike - premium
            if breakeven <= 0:
                return 0.0
            # P(S_T < breakeven)
            d = (np.log(spot / breakeven) + (self.risk_free_rate - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return 1 - norm.cdf(d)

    def _score_contract(
        self, greeks: Dict, pop: float, volume: int, oi: int,
        bid: float, ask: float, premium: float, pct_to_breakeven: float
    ) -> float:
        """
        Score an individual contract (0-100) for day/swing trading suitability.
        Factors:
        - Delta (want 0.30-0.50 for good leverage)
        - Probability of profit
        - Liquidity (volume, OI, bid-ask spread)
        - Theta decay cost
        - Distance to breakeven
        """
        score = 0.0

        # Delta sweet spot (0-25 points): 0.30-0.50 delta is ideal for day trades
        abs_delta = abs(greeks["delta"])
        if 0.30 <= abs_delta <= 0.50:
            score += 25
        elif 0.20 <= abs_delta <= 0.60:
            score += 15
        elif 0.10 <= abs_delta <= 0.70:
            score += 8

        # Probability of profit (0-25 points)
        score += pop * 25

        # Liquidity (0-25 points)
        liquidity_score = 0
        if volume > 1000:
            liquidity_score += 8
        elif volume > 100:
            liquidity_score += 4
        if oi > 5000:
            liquidity_score += 7
        elif oi > 500:
            liquidity_score += 3
        # Tight spread
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / premium if premium > 0 else 1
            if spread_pct < 0.05:
                liquidity_score += 10
            elif spread_pct < 0.15:
                liquidity_score += 6
            elif spread_pct < 0.30:
                liquidity_score += 3
        score += min(liquidity_score, 25)

        # Distance to breakeven (0-15 points): closer is better for day trades
        abs_pct = abs(pct_to_breakeven)
        if abs_pct < 0.01:
            score += 15
        elif abs_pct < 0.02:
            score += 12
        elif abs_pct < 0.03:
            score += 8
        elif abs_pct < 0.05:
            score += 4

        # Gamma bonus (0-10 points): high gamma = big delta changes = leverage
        if greeks["gamma"] > 0.05:
            score += 10
        elif greeks["gamma"] > 0.02:
            score += 5

        return score

    def _get_technical_bias(self, hist: pd.DataFrame) -> Dict:
        """Determine directional bias from simple technicals."""
        close = hist["Close"]

        # EMAs
        ema_9 = close.ewm(span=9).mean().iloc[-1]
        ema_21 = close.ewm(span=21).mean().iloc[-1]
        current = close.iloc[-1]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = (ema_12 - ema_26).iloc[-1]
        signal_line = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]

        # Determine bias
        bullish_signals = 0
        bearish_signals = 0

        if current > ema_9 > ema_21:
            bullish_signals += 2
        elif current < ema_9 < ema_21:
            bearish_signals += 2

        if rsi > 50:
            bullish_signals += 1
        elif rsi < 50:
            bearish_signals += 1

        if macd_line > signal_line:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            direction = "BULLISH"
        elif bearish_signals > bullish_signals:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "confidence": max(bullish_signals, bearish_signals) / 4.0,
            "rsi": round(rsi, 2),
            "macd_signal": "BULLISH" if macd_line > signal_line else "BEARISH",
            "ema_trend": "ABOVE" if current > ema_21 else "BELOW",
            "price_vs_ema9": round((current - ema_9) / ema_9 * 100, 2),
        }

    def _recommend_strategies(
        self,
        ticker: str,
        current_price: float,
        hist_vol: float,
        hist: pd.DataFrame,
        days_to_exp: int,
        affordable_calls: List[Dict],
        affordable_puts: List[Dict],
        raw_calls: pd.DataFrame,
        raw_puts: pd.DataFrame,
        expiration: str,
    ) -> List[Dict]:
        """
        Recommend specific strategies based on the analysis.
        Each strategy includes contracts, rationale, and risk parameters.
        """
        strategies = []
        bias = self._get_technical_bias(hist)

        # --- Strategy 1: Directional long call or put ---
        if bias["direction"] == "BULLISH" and affordable_calls:
            best = sorted(affordable_calls, key=lambda x: x["score"], reverse=True)[0]
            strategies.append({
                "name": "Long Call (Bullish Momentum)",
                "type": "LONG_CALL",
                "contracts": [{"action": "BUY", "type": "call", **best}],
                "rationale": (
                    f"{ticker} showing bullish bias (RSI {bias['rsi']}, "
                    f"MACD {bias['macd_signal']}, trend {bias['ema_trend']}). "
                    f"Delta {best['delta']} gives good directional exposure."
                ),
                "max_loss": best["total_cost"],
                "max_gain": "Unlimited",
                "breakeven": best["breakeven"],
                "ideal_exit": "Take profit at 50-100% gain on premium, or cut at 30-50% loss",
                "risk_level": "MODERATE",
            })

        if bias["direction"] == "BEARISH" and affordable_puts:
            best = sorted(affordable_puts, key=lambda x: x["score"], reverse=True)[0]
            strategies.append({
                "name": "Long Put (Bearish Momentum)",
                "type": "LONG_PUT",
                "contracts": [{"action": "BUY", "type": "put", **best}],
                "rationale": (
                    f"{ticker} showing bearish bias (RSI {bias['rsi']}, "
                    f"MACD {bias['macd_signal']}, trend {bias['ema_trend']}). "
                    f"Delta {best['delta']} gives good directional exposure."
                ),
                "max_loss": best["total_cost"],
                "max_gain": f"Up to ${round(best['breakeven'] * 100, 2)}",
                "breakeven": best["breakeven"],
                "ideal_exit": "Take profit at 50-100% gain on premium, or cut at 30-50% loss",
                "risk_level": "MODERATE",
            })

        # --- Strategy 2: ATM straddle for expected big move (if affordable) ---
        atm_strike = round(current_price)
        atm_call = next((c for c in affordable_calls if abs(c["strike"] - atm_strike) <= current_price * 0.02), None)
        atm_put = next((p for p in affordable_puts if abs(p["strike"] - atm_strike) <= current_price * 0.02), None)

        if atm_call and atm_put and (atm_call["total_cost"] + atm_put["total_cost"]) <= self.budget:
            total_cost = atm_call["total_cost"] + atm_put["total_cost"]
            strategies.append({
                "name": "Long Straddle (Big Move Expected)",
                "type": "LONG_STRADDLE",
                "contracts": [
                    {"action": "BUY", "type": "call", **atm_call},
                    {"action": "BUY", "type": "put", **atm_put},
                ],
                "rationale": (
                    f"Play both directions if expecting a big move. "
                    f"Total cost ${round(total_cost, 2)}. Stock needs to move "
                    f">{round((atm_call['premium'] + atm_put['premium']) / current_price * 100, 1)}% to profit."
                ),
                "max_loss": round(total_cost, 2),
                "max_gain": "Unlimited",
                "breakeven": f"Below ${round(atm_strike - atm_call['premium'] - atm_put['premium'], 2)} "
                             f"or above ${round(atm_strike + atm_call['premium'] + atm_put['premium'], 2)}",
                "ideal_exit": "Close when one side doubles, covering the other side's loss",
                "risk_level": "HIGH",
            })

        # --- Strategy 3: Debit spread (defined risk, lower cost) ---
        if bias["direction"] == "BULLISH" and len(affordable_calls) >= 2:
            sorted_calls = sorted(affordable_calls, key=lambda x: x["strike"])
            # Buy lower strike, sell higher strike
            for i, buy_leg in enumerate(sorted_calls[:-1]):
                for sell_leg in sorted_calls[i + 1:]:
                    if sell_leg["strike"] - buy_leg["strike"] <= current_price * 0.05:
                        net_debit = buy_leg["premium"] - sell_leg["premium"]
                        if net_debit > 0 and net_debit * 100 <= self.budget:
                            spread_width = sell_leg["strike"] - buy_leg["strike"]
                            max_gain = (spread_width - net_debit) * 100
                            strategies.append({
                                "name": "Bull Call Spread (Defined Risk)",
                                "type": "BULL_CALL_SPREAD",
                                "contracts": [
                                    {"action": "BUY", "type": "call", **buy_leg},
                                    {"action": "SELL", "type": "call", **sell_leg},
                                ],
                                "rationale": (
                                    f"Defined-risk bullish play. Buy ${buy_leg['strike']} call, "
                                    f"sell ${sell_leg['strike']} call. Net debit ${round(net_debit, 2)}/share. "
                                    f"Max gain ${round(max_gain, 2)} if stock above ${sell_leg['strike']} at expiry."
                                ),
                                "max_loss": round(net_debit * 100, 2),
                                "max_gain": round(max_gain, 2),
                                "breakeven": round(buy_leg["strike"] + net_debit, 2),
                                "ideal_exit": "Close at 50% of max gain or if thesis breaks",
                                "risk_level": "LOW",
                            })
                            break
                    break

        if bias["direction"] == "BEARISH" and len(affordable_puts) >= 2:
            sorted_puts = sorted(affordable_puts, key=lambda x: x["strike"], reverse=True)
            for i, buy_leg in enumerate(sorted_puts[:-1]):
                for sell_leg in sorted_puts[i + 1:]:
                    if buy_leg["strike"] - sell_leg["strike"] <= current_price * 0.05:
                        net_debit = buy_leg["premium"] - sell_leg["premium"]
                        if net_debit > 0 and net_debit * 100 <= self.budget:
                            spread_width = buy_leg["strike"] - sell_leg["strike"]
                            max_gain = (spread_width - net_debit) * 100
                            strategies.append({
                                "name": "Bear Put Spread (Defined Risk)",
                                "type": "BEAR_PUT_SPREAD",
                                "contracts": [
                                    {"action": "BUY", "type": "put", **buy_leg},
                                    {"action": "SELL", "type": "put", **sell_leg},
                                ],
                                "rationale": (
                                    f"Defined-risk bearish play. Buy ${buy_leg['strike']} put, "
                                    f"sell ${sell_leg['strike']} put. Net debit ${round(net_debit, 2)}/share. "
                                    f"Max gain ${round(max_gain, 2)} if stock below ${sell_leg['strike']} at expiry."
                                ),
                                "max_loss": round(net_debit * 100, 2),
                                "max_gain": round(max_gain, 2),
                                "breakeven": round(buy_leg["strike"] - net_debit, 2),
                                "ideal_exit": "Close at 50% of max gain or if thesis breaks",
                                "risk_level": "LOW",
                            })
                            break
                    break

        # --- Strategy 4: High-gamma scalp (very short-term) ---
        if days_to_exp <= 2:
            all_contracts = affordable_calls + affordable_puts
            high_gamma = sorted(all_contracts, key=lambda x: x.get("gamma", 0), reverse=True)
            if high_gamma:
                top = high_gamma[0]
                strategies.append({
                    "name": f"Gamma Scalp (0DTE/1DTE {top['type'].upper()})",
                    "type": "GAMMA_SCALP",
                    "contracts": [{"action": "BUY", **top}],
                    "rationale": (
                        f"High gamma ({top['gamma']}) means delta moves fast with price. "
                        f"Quick in-and-out on momentum. Theta decay is aggressive — "
                        f"exit same day."
                    ),
                    "max_loss": top["total_cost"],
                    "max_gain": "2-5x on a strong move",
                    "breakeven": top["breakeven"],
                    "ideal_exit": "Scalp 20-50% gain quickly, hard stop at 40% loss",
                    "risk_level": "HIGH",
                })

        return strategies
