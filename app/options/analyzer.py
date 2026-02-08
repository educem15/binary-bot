"""
Options Analyzer Module
Deep-dives into specific tickers to analyze options chains, Greeks,
probability of profit, and recommends optimal strategies with full trade plans.
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
        Returns Greeks, probability metrics, strategy recommendations, and trade plans.
        """
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

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

        # Options chain
        chain = stock.option_chain(target_exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        # Analyze chains
        analyzed_calls = self._analyze_chain(calls, current_price, hist_vol, years_to_exp, "call")
        analyzed_puts = self._analyze_chain(puts, current_price, hist_vol, years_to_exp, "put")

        # Filter to affordable
        affordable_calls = [c for c in analyzed_calls if c["total_cost"] <= self.budget]
        affordable_puts = [p for p in analyzed_puts if p["total_cost"] <= self.budget]

        # Technical bias with 200 EMA
        bias = self._get_technical_bias(hist)

        # Expected move
        expected_move = self._calculate_expected_move(current_price, hist_vol, days_to_exp)

        # Support/resistance
        sr_levels = self._calculate_support_resistance(hist, current_price)

        # Generate strategies with full trade plans
        strategies = self._recommend_strategies(
            ticker, current_price, hist_vol, hist, days_to_exp,
            affordable_calls, affordable_puts, calls, puts, target_exp,
            expected_move, sr_levels, bias
        )

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "historical_volatility": round(hist_vol, 4),
            "expiration": target_exp,
            "days_to_expiration": days_to_exp,
            "technical_bias": bias,
            "expected_move": expected_move,
            "support_resistance": sr_levels,
            "top_calls": sorted(affordable_calls, key=lambda x: x["score"], reverse=True)[:5],
            "top_puts": sorted(affordable_puts, key=lambda x: x["score"], reverse=True)[:5],
            "strategies": strategies,
        }

    # ── Expected Move ─────────────────────────────────────────────────

    def _calculate_expected_move(
        self, current_price: float, vol: float, days: int
    ) -> Dict:
        """Calculate the expected price move based on implied volatility."""
        daily_move = current_price * vol / np.sqrt(252)
        period_move = current_price * vol * np.sqrt(days / 252)

        return {
            "daily_expected_move": round(daily_move, 2),
            "daily_expected_move_pct": round(daily_move / current_price * 100, 2),
            "period_expected_move": round(period_move, 2),
            "period_expected_move_pct": round(period_move / current_price * 100, 2),
            "expected_range_high": round(current_price + period_move, 2),
            "expected_range_low": round(current_price - period_move, 2),
            "one_sd_up": round(current_price + period_move, 2),
            "one_sd_down": round(current_price - period_move, 2),
            "two_sd_up": round(current_price + 2 * period_move, 2),
            "two_sd_down": round(current_price - 2 * period_move, 2),
        }

    # ── Support / Resistance ──────────────────────────────────────────

    def _calculate_support_resistance(self, hist: pd.DataFrame, current_price: float) -> Dict:
        """Calculate support and resistance from pivots and recent price action."""
        recent = hist.tail(60)
        high = recent["High"]
        low = recent["Low"]
        close = recent["Close"]

        last_high = high.iloc[-1]
        last_low = low.iloc[-1]
        last_close = close.iloc[-1]
        pivot = (last_high + last_low + last_close) / 3

        r1 = 2 * pivot - last_low
        s1 = 2 * pivot - last_high
        r2 = pivot + (last_high - last_low)
        s2 = pivot - (last_high - last_low)

        return {
            "pivot": round(pivot, 2),
            "resistance_1": round(r1, 2),
            "resistance_2": round(r2, 2),
            "support_1": round(s1, 2),
            "support_2": round(s2, 2),
            "twenty_day_high": round(high.tail(20).max(), 2),
            "twenty_day_low": round(low.tail(20).min(), 2),
        }

    # ── Chain Analysis ────────────────────────────────────────────────

    def _analyze_chain(
        self, chain: pd.DataFrame, spot: float, vol: float, T: float, option_type: str,
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

            greeks = self._calculate_greeks(spot, strike, T, self.risk_free_rate, iv, option_type)
            pop = self._probability_of_profit(spot, strike, premium, T, iv, option_type)

            if option_type == "call":
                breakeven = strike + premium
                pct_to_breakeven = (breakeven - spot) / spot
            else:
                breakeven = strike - premium
                pct_to_breakeven = (spot - breakeven) / spot

            max_loss = total_cost

            # P&L scenarios at different price moves
            pnl_scenarios = self._calculate_pnl_scenarios(
                spot, strike, premium, T, iv, option_type
            )

            score = self._score_contract(
                greeks, pop, volume or 0, open_interest or 0,
                bid, ask, premium, pct_to_breakeven
            )

            mid_price = round((bid + ask) / 2, 2) if bid and ask and bid > 0 else premium

            results.append({
                "strike": strike,
                "premium": round(premium, 2),
                "mid_price": mid_price,
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
                "pnl_scenarios": pnl_scenarios,
                "score": round(score, 2),
                "type": option_type,
            })

        return results

    # ── P&L Scenarios ─────────────────────────────────────────────────

    def _calculate_pnl_scenarios(
        self, spot: float, strike: float, premium: float,
        T: float, iv: float, option_type: str
    ) -> List[Dict]:
        """Calculate P&L at different price move scenarios."""
        scenarios = []
        moves = [-5, -3, -2, -1, 0, 1, 2, 3, 5]

        for pct in moves:
            new_price = spot * (1 + pct / 100)

            # Intrinsic value at expiration
            if option_type == "call":
                intrinsic = max(new_price - strike, 0)
            else:
                intrinsic = max(strike - new_price, 0)

            pnl_per_share = intrinsic - premium
            pnl_total = pnl_per_share * 100
            pnl_pct = (pnl_per_share / premium * 100) if premium > 0 else 0

            # Estimated value at halfway to expiration (using BS approximation)
            T_half = T / 2 if T > 0 else 0.001
            if T_half > 0 and iv > 0:
                try:
                    if option_type == "call":
                        half_value = self._bs_price(new_price, strike, T_half, self.risk_free_rate, iv, "call")
                    else:
                        half_value = self._bs_price(new_price, strike, T_half, self.risk_free_rate, iv, "put")
                    intraday_pnl = (half_value - premium) * 100
                except Exception:
                    intraday_pnl = None
            else:
                intraday_pnl = None

            scenarios.append({
                "stock_move_pct": pct,
                "stock_price": round(new_price, 2),
                "option_value_at_exp": round(intrinsic, 2),
                "pnl_at_exp": round(pnl_total, 2),
                "pnl_pct_at_exp": round(pnl_pct, 1),
                "est_pnl_intraday": round(intraday_pnl, 2) if intraday_pnl is not None else None,
            })

        return scenarios

    def _bs_price(
        self, S: float, K: float, T: float, r: float, sigma: float, option_type: str
    ) -> float:
        """Black-Scholes option price."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # ── Greeks ────────────────────────────────────────────────────────

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

    # ── Probability of Profit ─────────────────────────────────────────

    def _probability_of_profit(
        self, spot: float, strike: float, premium: float,
        T: float, sigma: float, option_type: str
    ) -> float:
        """Estimate probability of profit at expiration."""
        if sigma <= 0 or T <= 0:
            return 0.0

        if option_type == "call":
            breakeven = strike + premium
            d = (np.log(spot / breakeven) + (self.risk_free_rate - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return norm.cdf(d)
        else:
            breakeven = strike - premium
            if breakeven <= 0:
                return 0.0
            d = (np.log(spot / breakeven) + (self.risk_free_rate - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return 1 - norm.cdf(d)

    # ── Contract Scoring ──────────────────────────────────────────────

    def _score_contract(
        self, greeks: Dict, pop: float, volume: int, oi: int,
        bid: float, ask: float, premium: float, pct_to_breakeven: float
    ) -> float:
        """Score an individual contract (0-100) for day/swing suitability."""
        score = 0.0

        # Delta sweet spot (0-25 pts)
        abs_delta = abs(greeks["delta"])
        if 0.30 <= abs_delta <= 0.50:
            score += 25
        elif 0.20 <= abs_delta <= 0.60:
            score += 15
        elif 0.10 <= abs_delta <= 0.70:
            score += 8

        # Probability of profit (0-25 pts)
        score += pop * 25

        # Liquidity (0-25 pts)
        liq = 0
        if volume > 1000:
            liq += 8
        elif volume > 100:
            liq += 4
        if oi > 5000:
            liq += 7
        elif oi > 500:
            liq += 3
        if bid > 0 and ask > 0 and premium > 0:
            spread_pct = (ask - bid) / premium
            if spread_pct < 0.05:
                liq += 10
            elif spread_pct < 0.15:
                liq += 6
            elif spread_pct < 0.30:
                liq += 3
        score += min(liq, 25)

        # Distance to breakeven (0-15 pts)
        abs_pct = abs(pct_to_breakeven)
        if abs_pct < 0.01:
            score += 15
        elif abs_pct < 0.02:
            score += 12
        elif abs_pct < 0.03:
            score += 8
        elif abs_pct < 0.05:
            score += 4

        # Gamma bonus (0-10 pts)
        if greeks["gamma"] > 0.05:
            score += 10
        elif greeks["gamma"] > 0.02:
            score += 5

        return score

    # ── Technical Bias (with 200 EMA) ─────────────────────────────────

    def _get_technical_bias(self, hist: pd.DataFrame) -> Dict:
        """Determine directional bias including 200 EMA analysis."""
        close = hist["Close"]
        current = close.iloc[-1]

        # EMAs
        ema_9 = close.ewm(span=9).mean().iloc[-1]
        ema_21 = close.ewm(span=21).mean().iloc[-1]
        ema_50 = close.ewm(span=50).mean().iloc[-1]
        ema_200 = close.ewm(span=200).mean().iloc[-1] if len(close) >= 200 else None

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
        macd_histogram = macd_line - signal_line

        # VWAP approximation (volume-weighted close)
        if "Volume" in hist.columns:
            recent = hist.tail(20)
            vwap = (recent["Close"] * recent["Volume"]).sum() / recent["Volume"].sum()
        else:
            vwap = current

        # 200 EMA proximity
        ema_200_distance_pct = None
        testing_200_ema = False
        ema_200_position = None
        if ema_200:
            ema_200_distance_pct = round((current - ema_200) / ema_200 * 100, 2)
            testing_200_ema = abs(ema_200_distance_pct) <= 1.0
            ema_200_position = "ABOVE" if current > ema_200 else "BELOW"

        # Scoring
        bullish = 0
        bearish = 0

        if current > ema_9 > ema_21:
            bullish += 2
        elif current < ema_9 < ema_21:
            bearish += 2

        if rsi > 50:
            bullish += 1
        elif rsi < 50:
            bearish += 1

        if macd_line > signal_line:
            bullish += 1
        else:
            bearish += 1

        if macd_histogram > 0:
            bullish += 1
        else:
            bearish += 1

        if ema_200 and current > ema_200:
            bullish += 1
        elif ema_200:
            bearish += 1

        if current > vwap:
            bullish += 1
        else:
            bearish += 1

        total_signals = bullish + bearish
        if bullish > bearish:
            direction = "BULLISH"
        elif bearish > bullish:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "confidence": round(max(bullish, bearish) / max(total_signals, 1), 2),
            "bullish_signals": bullish,
            "bearish_signals": bearish,
            "rsi": round(rsi, 2),
            "macd_signal": "BULLISH" if macd_line > signal_line else "BEARISH",
            "macd_histogram": round(macd_histogram, 4),
            "ema_trend": "ABOVE" if current > ema_21 else "BELOW",
            "price_vs_ema9": round((current - ema_9) / ema_9 * 100, 2),
            "ema_200_value": round(ema_200, 2) if ema_200 else None,
            "ema_200_distance_pct": ema_200_distance_pct,
            "testing_200_ema": testing_200_ema,
            "ema_200_position": ema_200_position,
            "vwap": round(vwap, 2),
            "above_vwap": current > vwap,
        }

    # ── Strategy Recommendations with Full Trade Plans ────────────────

    def _recommend_strategies(
        self, ticker: str, current_price: float, hist_vol: float,
        hist: pd.DataFrame, days_to_exp: int,
        affordable_calls: List[Dict], affordable_puts: List[Dict],
        raw_calls: pd.DataFrame, raw_puts: pd.DataFrame, expiration: str,
        expected_move: Dict, sr_levels: Dict, bias: Dict,
    ) -> List[Dict]:
        """Recommend strategies with complete trade plans."""
        strategies = []

        # ── Strategy 1: Directional (aligned with bias) ──
        if bias["direction"] == "BULLISH" and affordable_calls:
            best = sorted(affordable_calls, key=lambda x: x["score"], reverse=True)[0]
            trade_plan = self._build_trade_plan(
                best, current_price, expected_move, sr_levels, "call", bias
            )
            strategies.append({
                "name": "Long Call (Bullish Momentum)",
                "type": "LONG_CALL",
                "contracts": [{"action": "BUY", "type": "call", **best}],
                "trade_plan": trade_plan,
                "rationale": (
                    f"{ticker} showing bullish bias ({bias['bullish_signals']}/{bias['bullish_signals']+bias['bearish_signals']} signals). "
                    f"RSI {bias['rsi']}, MACD {bias['macd_signal']}, "
                    f"{'above' if bias.get('above_vwap') else 'below'} VWAP. "
                    f"{'Testing 200 EMA support — potential bounce play. ' if bias.get('testing_200_ema') and bias.get('ema_200_position') == 'ABOVE' else ''}"
                    f"Delta {best['delta']} provides ${round(abs(best['delta']) * 100, 0)}/pt exposure."
                ),
                "max_loss": best["total_cost"],
                "max_gain": "Unlimited",
                "breakeven": best["breakeven"],
                "risk_level": "MODERATE",
            })

        if bias["direction"] == "BEARISH" and affordable_puts:
            best = sorted(affordable_puts, key=lambda x: x["score"], reverse=True)[0]
            trade_plan = self._build_trade_plan(
                best, current_price, expected_move, sr_levels, "put", bias
            )
            strategies.append({
                "name": "Long Put (Bearish Momentum)",
                "type": "LONG_PUT",
                "contracts": [{"action": "BUY", "type": "put", **best}],
                "trade_plan": trade_plan,
                "rationale": (
                    f"{ticker} showing bearish bias ({bias['bearish_signals']}/{bias['bullish_signals']+bias['bearish_signals']} signals). "
                    f"RSI {bias['rsi']}, MACD {bias['macd_signal']}, "
                    f"{'below' if not bias.get('above_vwap') else 'above'} VWAP. "
                    f"{'Rejected at 200 EMA — potential breakdown. ' if bias.get('testing_200_ema') and bias.get('ema_200_position') == 'BELOW' else ''}"
                    f"Delta {best['delta']} provides ${round(abs(best['delta']) * 100, 0)}/pt exposure."
                ),
                "max_loss": best["total_cost"],
                "max_gain": f"Up to ${round(best['breakeven'] * 100, 2)}",
                "breakeven": best["breakeven"],
                "risk_level": "MODERATE",
            })

        # ── Strategy 2: 200 EMA Bounce/Rejection Play ──
        if bias.get("testing_200_ema") or (bias.get("ema_200_distance_pct") and abs(bias["ema_200_distance_pct"]) <= 2.0):
            if bias.get("ema_200_position") == "ABOVE" and affordable_calls:
                # Bullish bounce off 200 EMA
                best = sorted(affordable_calls, key=lambda x: x["score"], reverse=True)[0]
                trade_plan = self._build_trade_plan(
                    best, current_price, expected_move, sr_levels, "call", bias
                )
                trade_plan["entry_trigger"] = f"Enter on bounce confirmation above 200 EMA (${bias.get('ema_200_value', 'N/A')})"
                trade_plan["stop_loss_price"] = round(bias.get("ema_200_value", current_price) * 0.99, 2)
                strategies.append({
                    "name": "200 EMA Bounce (Bullish)",
                    "type": "EMA_200_BOUNCE",
                    "contracts": [{"action": "BUY", "type": "call", **best}],
                    "trade_plan": trade_plan,
                    "rationale": (
                        f"{ticker} testing 200 EMA at ${bias.get('ema_200_value')}. "
                        f"Price is {bias.get('ema_200_distance_pct')}% from 200 EMA. "
                        f"Historically a strong support level. Buy call on bounce confirmation."
                    ),
                    "max_loss": best["total_cost"],
                    "max_gain": "Unlimited",
                    "breakeven": best["breakeven"],
                    "risk_level": "MODERATE",
                })
            elif bias.get("ema_200_position") == "BELOW" and affordable_puts:
                # Bearish rejection at 200 EMA
                best = sorted(affordable_puts, key=lambda x: x["score"], reverse=True)[0]
                trade_plan = self._build_trade_plan(
                    best, current_price, expected_move, sr_levels, "put", bias
                )
                trade_plan["entry_trigger"] = f"Enter on rejection confirmation below 200 EMA (${bias.get('ema_200_value', 'N/A')})"
                trade_plan["stop_loss_price"] = round(bias.get("ema_200_value", current_price) * 1.01, 2)
                strategies.append({
                    "name": "200 EMA Rejection (Bearish)",
                    "type": "EMA_200_REJECTION",
                    "contracts": [{"action": "BUY", "type": "put", **best}],
                    "trade_plan": trade_plan,
                    "rationale": (
                        f"{ticker} testing 200 EMA resistance at ${bias.get('ema_200_value')}. "
                        f"Price is {bias.get('ema_200_distance_pct')}% from 200 EMA. "
                        f"Failure to break above = bearish continuation."
                    ),
                    "max_loss": best["total_cost"],
                    "max_gain": f"Up to ${round(best['breakeven'] * 100, 2)}",
                    "breakeven": best["breakeven"],
                    "risk_level": "MODERATE",
                })

        # ── Strategy 3: ATM Straddle (big move expected) ──
        atm_strike = round(current_price)
        atm_call = next((c for c in affordable_calls if abs(c["strike"] - atm_strike) <= current_price * 0.02), None)
        atm_put = next((p for p in affordable_puts if abs(p["strike"] - atm_strike) <= current_price * 0.02), None)

        if atm_call and atm_put and (atm_call["total_cost"] + atm_put["total_cost"]) <= self.budget:
            total_cost = atm_call["total_cost"] + atm_put["total_cost"]
            total_premium = atm_call["premium"] + atm_put["premium"]
            move_needed_pct = total_premium / current_price * 100
            strategies.append({
                "name": "Long Straddle (Big Move Expected)",
                "type": "LONG_STRADDLE",
                "contracts": [
                    {"action": "BUY", "type": "call", **atm_call},
                    {"action": "BUY", "type": "put", **atm_put},
                ],
                "trade_plan": {
                    "entry_price": f"Call @ ${atm_call['mid_price']} + Put @ ${atm_put['mid_price']}",
                    "total_cost": round(total_cost, 2),
                    "entry_trigger": "Enter before expected catalyst or high-volatility event",
                    "profit_target_1": f"Close winning leg when it covers total premium (${round(total_premium, 2)}/share move needed)",
                    "profit_target_2": f"Close both legs when net +50% (${round(total_cost * 0.5, 2)} profit)",
                    "stop_loss": f"Close if stock stagnates — time decay eats both legs at ${round(abs(atm_call['theta'] + atm_put['theta']), 2)}/day",
                    "stop_loss_price": f"Cut at 30% loss (${round(total_cost * 0.3, 2)})",
                    "move_needed_pct": round(move_needed_pct, 1),
                    "expected_daily_move_pct": expected_move["daily_expected_move_pct"],
                    "time_horizon": "1-3 days max, theta kills this position",
                },
                "rationale": (
                    f"Play both directions. Needs >{move_needed_pct:.1f}% move to profit. "
                    f"Expected daily move is {expected_move['daily_expected_move_pct']}%. "
                    f"{'Good ratio — expected move exceeds cost.' if expected_move['daily_expected_move_pct'] > move_needed_pct else 'Tight — needs above-average move.'}"
                ),
                "max_loss": round(total_cost, 2),
                "max_gain": "Unlimited",
                "breakeven": f"${round(atm_strike - total_premium, 2)} / ${round(atm_strike + total_premium, 2)}",
                "risk_level": "HIGH",
            })

        # ── Strategy 4: Debit Spread (defined risk) ──
        if bias["direction"] == "BULLISH" and len(affordable_calls) >= 2:
            sorted_calls = sorted(affordable_calls, key=lambda x: x["strike"])
            for i, buy_leg in enumerate(sorted_calls[:-1]):
                for sell_leg in sorted_calls[i + 1:]:
                    if sell_leg["strike"] - buy_leg["strike"] <= current_price * 0.05:
                        net_debit = buy_leg["premium"] - sell_leg["premium"]
                        if net_debit > 0 and net_debit * 100 <= self.budget:
                            spread_width = sell_leg["strike"] - buy_leg["strike"]
                            max_gain = (spread_width - net_debit) * 100
                            risk_reward_ratio = max_gain / (net_debit * 100) if net_debit > 0 else 0
                            strategies.append({
                                "name": "Bull Call Spread (Defined Risk)",
                                "type": "BULL_CALL_SPREAD",
                                "contracts": [
                                    {"action": "BUY", "type": "call", **buy_leg},
                                    {"action": "SELL", "type": "call", **sell_leg},
                                ],
                                "trade_plan": {
                                    "entry_price": f"Net debit ${round(net_debit, 2)}/share (${round(net_debit * 100, 2)} total)",
                                    "total_cost": round(net_debit * 100, 2),
                                    "entry_trigger": "Enter on pullback to support or momentum confirmation",
                                    "profit_target_1": f"Close at 50% max gain (${round(max_gain * 0.5, 2)})",
                                    "profit_target_2": f"Hold for full ${round(max_gain, 2)} if above ${sell_leg['strike']} at expiry",
                                    "stop_loss": f"Close if stock breaks below ${round(buy_leg['strike'] - net_debit, 2)}",
                                    "stop_loss_price": round(buy_leg["strike"] - net_debit, 2),
                                    "risk_reward_ratio": f"{risk_reward_ratio:.1f}:1",
                                    "time_horizon": f"Hold to expiration or close at target",
                                },
                                "rationale": (
                                    f"Defined-risk: Buy ${buy_leg['strike']}C, sell ${sell_leg['strike']}C. "
                                    f"Risk ${round(net_debit * 100, 2)} to make ${round(max_gain, 2)} "
                                    f"({risk_reward_ratio:.1f}:1). Need stock above ${round(buy_leg['strike'] + net_debit, 2)} to profit."
                                ),
                                "max_loss": round(net_debit * 100, 2),
                                "max_gain": round(max_gain, 2),
                                "breakeven": round(buy_leg["strike"] + net_debit, 2),
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
                            risk_reward_ratio = max_gain / (net_debit * 100) if net_debit > 0 else 0
                            strategies.append({
                                "name": "Bear Put Spread (Defined Risk)",
                                "type": "BEAR_PUT_SPREAD",
                                "contracts": [
                                    {"action": "BUY", "type": "put", **buy_leg},
                                    {"action": "SELL", "type": "put", **sell_leg},
                                ],
                                "trade_plan": {
                                    "entry_price": f"Net debit ${round(net_debit, 2)}/share (${round(net_debit * 100, 2)} total)",
                                    "total_cost": round(net_debit * 100, 2),
                                    "entry_trigger": "Enter on rejection at resistance or breakdown confirmation",
                                    "profit_target_1": f"Close at 50% max gain (${round(max_gain * 0.5, 2)})",
                                    "profit_target_2": f"Hold for full ${round(max_gain, 2)} if below ${sell_leg['strike']} at expiry",
                                    "stop_loss": f"Close if stock breaks above ${round(buy_leg['strike'] + net_debit, 2)}",
                                    "stop_loss_price": round(buy_leg["strike"] + net_debit, 2),
                                    "risk_reward_ratio": f"{risk_reward_ratio:.1f}:1",
                                    "time_horizon": f"Hold to expiration or close at target",
                                },
                                "rationale": (
                                    f"Defined-risk: Buy ${buy_leg['strike']}P, sell ${sell_leg['strike']}P. "
                                    f"Risk ${round(net_debit * 100, 2)} to make ${round(max_gain, 2)} "
                                    f"({risk_reward_ratio:.1f}:1). Need stock below ${round(buy_leg['strike'] - net_debit, 2)} to profit."
                                ),
                                "max_loss": round(net_debit * 100, 2),
                                "max_gain": round(max_gain, 2),
                                "breakeven": round(buy_leg["strike"] - net_debit, 2),
                                "risk_level": "LOW",
                            })
                            break
                    break

        # ── Strategy 5: Gamma Scalp (0DTE/1DTE) ──
        if days_to_exp <= 2:
            all_contracts = affordable_calls + affordable_puts
            high_gamma = sorted(all_contracts, key=lambda x: x.get("gamma", 0), reverse=True)
            if high_gamma:
                top = high_gamma[0]
                trade_plan = self._build_trade_plan(
                    top, current_price, expected_move, sr_levels, top["type"], bias
                )
                trade_plan["time_horizon"] = "Minutes to hours — same-day exit mandatory"
                trade_plan["entry_trigger"] = "Enter on momentum surge with volume confirmation"
                strategies.append({
                    "name": f"Gamma Scalp (0DTE/1DTE {top['type'].upper()})",
                    "type": "GAMMA_SCALP",
                    "contracts": [{"action": "BUY", **top}],
                    "trade_plan": trade_plan,
                    "rationale": (
                        f"High gamma ({top['gamma']}) = delta moves fast. "
                        f"A ${expected_move['daily_expected_move']:.2f} move today makes delta shift rapidly. "
                        f"Quick in-and-out. Theta is ${abs(top['theta']):.2f}/day — exit same session."
                    ),
                    "max_loss": top["total_cost"],
                    "max_gain": "2-5x on a strong move",
                    "breakeven": top["breakeven"],
                    "risk_level": "HIGH",
                })

        return strategies

    # ── Trade Plan Builder ────────────────────────────────────────────

    def _build_trade_plan(
        self, contract: Dict, current_price: float,
        expected_move: Dict, sr_levels: Dict,
        option_type: str, bias: Dict
    ) -> Dict:
        """Build a complete trade plan for a single contract."""
        premium = contract["premium"]
        strike = contract["strike"]
        total_cost = contract["total_cost"]
        mid_price = contract.get("mid_price", premium)

        # Entry
        entry_price = mid_price
        entry_total = round(entry_price * 100, 2)

        # Targets based on premium % gain
        target_50_pct = round(premium * 1.5, 2)
        target_100_pct = round(premium * 2.0, 2)
        profit_50 = round((target_50_pct - premium) * 100, 2)
        profit_100 = round((target_100_pct - premium) * 100, 2)

        # Stop loss
        stop_loss_premium = round(premium * 0.60, 2)  # 40% loss on premium
        stop_loss_dollar = round((premium - stop_loss_premium) * 100, 2)

        # Price targets based on S/R and expected move
        if option_type == "call":
            price_target_1 = sr_levels.get("resistance_1", round(current_price * 1.02, 2))
            price_target_2 = sr_levels.get("resistance_2", round(current_price * 1.04, 2))
            stop_trigger_price = sr_levels.get("support_1", round(current_price * 0.98, 2))
        else:
            price_target_1 = sr_levels.get("support_1", round(current_price * 0.98, 2))
            price_target_2 = sr_levels.get("support_2", round(current_price * 0.96, 2))
            stop_trigger_price = sr_levels.get("resistance_1", round(current_price * 1.02, 2))

        return {
            "option_to_buy": f"{'CALL' if option_type == 'call' else 'PUT'} ${strike} exp {contract.get('expiration', 'nearest')}",
            "entry_price": f"${entry_price} (mid) — limit order at ${mid_price}",
            "total_cost": entry_total,
            "contracts_qty": 1,
            "profit_target_1": f"Sell at ${target_50_pct} (+50% = +${profit_50})",
            "profit_target_2": f"Sell at ${target_100_pct} (+100% = +${profit_100})",
            "stock_price_target_1": price_target_1,
            "stock_price_target_2": price_target_2,
            "stop_loss": f"Sell at ${stop_loss_premium} (-40% = -${stop_loss_dollar})",
            "stop_loss_price": stop_trigger_price,
            "risk_amount": stop_loss_dollar,
            "reward_amount": profit_50,
            "risk_reward_ratio": f"{round(profit_50 / stop_loss_dollar, 1) if stop_loss_dollar > 0 else 'N/A'}:1",
            "time_horizon": "Same day (day trade) or 1-3 days (swing)",
            "entry_trigger": (
                f"Enter when stock confirms direction "
                f"{'above VWAP $' + str(bias.get('vwap', '')) if option_type == 'call' else 'below VWAP $' + str(bias.get('vwap', ''))}"
            ),
            "daily_theta_cost": f"${abs(contract.get('theta', 0)) * 100:.2f}/day",
            "what_to_expect": (
                f"Expected daily move: ${expected_move['daily_expected_move']} "
                f"({expected_move['daily_expected_move_pct']}%). "
                f"Delta {contract['delta']} = option moves ~${abs(contract['delta']):.2f} per $1 stock move. "
                f"Gamma {contract['gamma']} = delta accelerates on big moves."
            ),
        }
