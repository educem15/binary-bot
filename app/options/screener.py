"""
Options Screener Module
Scans the market for high-potential options trades within a given budget.
Includes squeeze detection, 200 EMA test, unusual options activity, and more.
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
        self.max_premium = budget / 100
        self.watchlist = watchlist or DEFAULT_WATCHLIST

    def scan_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Analyze a single ticker for options potential.
        Returns a summary dict or None if the ticker doesn't qualify.
        """
        try:
            stock = yf.Ticker(ticker)
            # Fetch 1 year of data for 200 EMA calculation
            hist = stock.history(period="1y")

            if hist.empty or len(hist) < 20:
                return None

            info = stock.info or {}
            current_price = hist["Close"].iloc[-1]

            # Beta
            beta = info.get("beta", None)

            # Historical volatility (20-day)
            returns = hist["Close"].pct_change().dropna()
            hist_vol_20d = returns.std() * np.sqrt(252) if len(returns) >= 5 else 0

            # Average daily volume
            avg_volume = hist["Volume"].tail(20).mean()

            # Price momentum
            price_5d_ago = hist["Close"].iloc[-6] if len(hist) >= 6 else hist["Close"].iloc[0]
            momentum_5d = (current_price - price_5d_ago) / price_5d_ago
            price_1d_ago = hist["Close"].iloc[-2] if len(hist) >= 2 else current_price
            momentum_1d = (current_price - price_1d_ago) / price_1d_ago

            # ATR
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

            # ── 200 EMA Analysis ──
            ema_200_data = self._analyze_ema200(hist, current_price)

            # ── Squeeze Detection ──
            squeeze_data = self._detect_squeeze(hist, info, current_price)

            # ── Unusual Options Activity ──
            try:
                expirations = stock.options
            except Exception:
                expirations = []

            if not expirations:
                return None

            # Find expiration buckets
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

            # Get IV data and unusual activity
            target_exps = near_term_exps or swing_exps
            iv_data = self._get_iv_data(stock, target_exps, current_price)
            unusual_activity = self._detect_unusual_options_activity(
                stock, target_exps, current_price
            )

            # ── Earnings Proximity ──
            earnings_near = self._check_earnings_proximity(info)

            # ── Support / Resistance ──
            sr_levels = self._calculate_support_resistance(hist, current_price)

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
                # New fields
                "ema_200": ema_200_data,
                "squeeze": squeeze_data,
                "unusual_options_activity": unusual_activity,
                "earnings_within_7d": earnings_near,
                "support_resistance": sr_levels,
            }

        except Exception as e:
            logger.warning(f"Error scanning {ticker}: {e}")
            return None

    # ── 200 EMA Analysis ──────────────────────────────────────────────

    def _analyze_ema200(self, hist: pd.DataFrame, current_price: float) -> Dict:
        """
        Analyze stock's relationship with the 200 EMA.
        Detects tests, bounces, and breakdowns.
        """
        close = hist["Close"]

        if len(close) < 200:
            # Not enough data for 200 EMA, use what we have
            ema_val = close.ewm(span=min(len(close), 200)).mean().iloc[-1]
            return {
                "ema_value": round(ema_val, 2),
                "distance_pct": round((current_price - ema_val) / ema_val * 100, 2),
                "position": "ABOVE" if current_price > ema_val else "BELOW",
                "testing": False,
                "touch_count_30d": 0,
                "bounce_pattern": None,
                "insufficient_data": True,
            }

        ema_200 = close.ewm(span=200).mean()
        ema_val = ema_200.iloc[-1]
        distance_pct = (current_price - ema_val) / ema_val * 100

        # Check if currently testing (within 1% of 200 EMA)
        testing = abs(distance_pct) <= 1.0

        # Count touches in last 30 days (price came within 0.5% of 200 EMA)
        recent = hist.tail(30)
        recent_ema = ema_200.tail(30)
        touch_threshold = 0.005  # 0.5%
        touches = 0
        touch_dates = []
        for i in range(len(recent)):
            price = recent["Close"].iloc[i]
            ema = recent_ema.iloc[i]
            if abs(price - ema) / ema <= touch_threshold:
                touches += 1
                touch_dates.append(str(recent.index[i].date()))

            # Also check if low touched from above or high touched from below
            if recent["Low"].iloc[i] <= ema <= recent["High"].iloc[i]:
                if str(recent.index[i].date()) not in touch_dates:
                    touches += 1
                    touch_dates.append(str(recent.index[i].date()))

        # Determine bounce pattern
        bounce_pattern = None
        if touches >= 1:
            # Check if price bounced up (bullish) or rejected down (bearish)
            prices_after_touch = close.tail(5)
            if len(prices_after_touch) >= 3:
                if prices_after_touch.iloc[-1] > ema_val and current_price > ema_val:
                    bounce_pattern = "BULLISH_BOUNCE"
                elif prices_after_touch.iloc[-1] < ema_val and current_price < ema_val:
                    bounce_pattern = "BEARISH_REJECTION"
                elif current_price > ema_val:
                    bounce_pattern = "HOLDING_ABOVE"
                else:
                    bounce_pattern = "HOLDING_BELOW"

        # Trend of 200 EMA itself (rising or falling)
        ema_slope = (ema_200.iloc[-1] - ema_200.iloc[-20]) / ema_200.iloc[-20] * 100

        return {
            "ema_value": round(ema_val, 2),
            "distance_pct": round(distance_pct, 2),
            "position": "ABOVE" if current_price > ema_val else "BELOW",
            "testing": testing,
            "touch_count_30d": touches,
            "recent_touch_dates": touch_dates[-5:],  # Last 5 touch dates
            "bounce_pattern": bounce_pattern,
            "ema_slope_20d": round(ema_slope, 2),
            "ema_trending": "RISING" if ema_slope > 0.5 else "FALLING" if ema_slope < -0.5 else "FLAT",
            "insufficient_data": False,
        }

    # ── Squeeze Detection ─────────────────────────────────────────────

    def _detect_squeeze(self, hist: pd.DataFrame, info: Dict, current_price: float) -> Dict:
        """
        Detect short squeeze potential using multiple signals:
        - Bollinger Band squeeze (volatility compression)
        - Volume surge detection
        - Short interest data from Yahoo Finance
        - Price compression patterns
        """
        close = hist["Close"]
        volume = hist["Volume"]

        # ── Bollinger Band Squeeze ──
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_width = ((bb_upper - bb_lower) / sma_20).dropna()

        # Squeeze = BB width at multi-week low (compression)
        if len(bb_width) >= 20:
            current_bb_width = bb_width.iloc[-1]
            bb_width_percentile = (bb_width < current_bb_width).sum() / len(bb_width)
            bb_squeeze = bb_width_percentile < 0.20  # Bottom 20% = squeeze
        else:
            current_bb_width = 0
            bb_width_percentile = 0.5
            bb_squeeze = False

        # ── Keltner Channel Squeeze (TTM Squeeze) ──
        # When BB is inside Keltner, it's a squeeze
        tr = pd.concat([
            hist["High"] - hist["Low"],
            (hist["High"] - close.shift(1)).abs(),
            (hist["Low"] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_20 = tr.rolling(20).mean()
        kc_upper = sma_20 + 1.5 * atr_20
        kc_lower = sma_20 - 1.5 * atr_20

        ttm_squeeze = False
        squeeze_firing = False
        if len(bb_upper) >= 20 and not bb_upper.isna().iloc[-1]:
            ttm_squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
            # Check if squeeze just released (was squeezing, now not)
            if len(bb_upper) >= 21:
                was_squeezing = (bb_upper.iloc[-2] < kc_upper.iloc[-2]) and (bb_lower.iloc[-2] > kc_lower.iloc[-2])
                squeeze_firing = was_squeezing and not ttm_squeeze

        # ── Volume Surge ──
        vol_sma_20 = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_sma_20.iloc[-1] if vol_sma_20.iloc[-1] > 0 else 1
        volume_surge = vol_ratio > 2.0

        # ── Short Interest (from Yahoo Finance info) ──
        short_pct_float = info.get("shortPercentOfFloat", None)
        if short_pct_float:
            short_pct_float = round(short_pct_float * 100, 2) if short_pct_float < 1 else round(short_pct_float, 2)
        shares_short = info.get("sharesShort", None)
        float_shares = info.get("floatShares", None)
        short_ratio = info.get("shortRatio", None)  # Days to cover

        high_short_interest = False
        if short_pct_float and short_pct_float > 15:
            high_short_interest = True
        elif short_ratio and short_ratio > 5:
            high_short_interest = True

        # ── Composite Squeeze Score (0-100) ──
        squeeze_score = 0
        if bb_squeeze:
            squeeze_score += 20
        if ttm_squeeze:
            squeeze_score += 25
        if squeeze_firing:
            squeeze_score += 15  # Squeeze just fired = immediate opportunity
        if volume_surge:
            squeeze_score += 15
        if high_short_interest:
            squeeze_score += 25
            if short_pct_float and short_pct_float > 25:
                squeeze_score += 10  # Extreme short interest bonus

        return {
            "squeeze_score": min(squeeze_score, 100),
            "bb_squeeze": bb_squeeze,
            "bb_width_percentile": round(bb_width_percentile * 100, 1),
            "ttm_squeeze": ttm_squeeze,
            "squeeze_firing": squeeze_firing,
            "volume_surge": volume_surge,
            "volume_ratio": round(vol_ratio, 2),
            "short_pct_float": short_pct_float,
            "shares_short": shares_short,
            "short_ratio_days": round(short_ratio, 1) if short_ratio else None,
            "high_short_interest": high_short_interest,
        }

    # ── Unusual Options Activity ──────────────────────────────────────

    def _detect_unusual_options_activity(
        self, stock, expirations: List[str], current_price: float
    ) -> Dict:
        """Detect unusual options activity by comparing volume to open interest."""
        if not expirations:
            return {"unusual": False, "signals": []}

        try:
            exp = expirations[0]
            chain = stock.option_chain(exp)
            calls = chain.calls
            puts = chain.puts

            signals = []
            total_call_vol = 0
            total_put_vol = 0
            total_call_oi = 0
            total_put_oi = 0

            # Check calls for unusual activity
            for _, row in calls.iterrows():
                vol = row.get("volume", 0)
                oi = row.get("openInterest", 0)
                if pd.isna(vol):
                    vol = 0
                if pd.isna(oi):
                    oi = 0
                total_call_vol += vol
                total_call_oi += oi

                if oi > 0 and vol > 0:
                    ratio = vol / oi
                    if ratio > 3.0 and vol > 500:
                        signals.append({
                            "type": "CALL",
                            "strike": row["strike"],
                            "volume": int(vol),
                            "open_interest": int(oi),
                            "vol_oi_ratio": round(ratio, 1),
                            "premium": round(row.get("lastPrice", 0), 2),
                        })

            # Check puts
            for _, row in puts.iterrows():
                vol = row.get("volume", 0)
                oi = row.get("openInterest", 0)
                if pd.isna(vol):
                    vol = 0
                if pd.isna(oi):
                    oi = 0
                total_put_vol += vol
                total_put_oi += oi

                if oi > 0 and vol > 0:
                    ratio = vol / oi
                    if ratio > 3.0 and vol > 500:
                        signals.append({
                            "type": "PUT",
                            "strike": row["strike"],
                            "volume": int(vol),
                            "open_interest": int(oi),
                            "vol_oi_ratio": round(ratio, 1),
                            "premium": round(row.get("lastPrice", 0), 2),
                        })

            # Put/Call ratio
            pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0

            # Sort signals by volume/OI ratio
            signals.sort(key=lambda x: x["vol_oi_ratio"], reverse=True)

            return {
                "unusual": len(signals) > 0,
                "signal_count": len(signals),
                "top_signals": signals[:5],
                "total_call_volume": int(total_call_vol),
                "total_put_volume": int(total_put_vol),
                "put_call_ratio": round(pc_ratio, 2),
                "sentiment": (
                    "VERY_BEARISH" if pc_ratio > 1.5
                    else "BEARISH" if pc_ratio > 1.0
                    else "NEUTRAL" if pc_ratio > 0.7
                    else "BULLISH" if pc_ratio > 0.4
                    else "VERY_BULLISH"
                ),
            }

        except Exception as e:
            logger.warning(f"Error detecting unusual activity: {e}")
            return {"unusual": False, "signals": []}

    # ── Earnings Proximity ────────────────────────────────────────────

    def _check_earnings_proximity(self, info: Dict) -> bool:
        """Check if earnings are within 7 days."""
        try:
            # yfinance stores earnings dates in various formats
            earnings_ts = info.get("mostRecentQuarter")
            if not earnings_ts:
                return False
            # Check for upcoming earnings (this is approximate from Yahoo data)
            # The "earningsTimestamp" field if available
            next_earnings = info.get("earningsTimestamp")
            if next_earnings:
                earnings_date = datetime.fromtimestamp(next_earnings)
                days_away = (earnings_date - datetime.now()).days
                return 0 <= days_away <= 7
        except Exception:
            pass
        return False

    # ── Support / Resistance ──────────────────────────────────────────

    def _calculate_support_resistance(self, hist: pd.DataFrame, current_price: float) -> Dict:
        """Calculate key support and resistance levels from price pivots."""
        recent = hist.tail(60)
        high = recent["High"]
        low = recent["Low"]
        close = recent["Close"]

        # Pivot points from last session
        last_high = high.iloc[-1]
        last_low = low.iloc[-1]
        last_close = close.iloc[-1]
        pivot = (last_high + last_low + last_close) / 3

        r1 = 2 * pivot - last_low
        s1 = 2 * pivot - last_high
        r2 = pivot + (last_high - last_low)
        s2 = pivot - (last_high - last_low)

        # Recent swing highs/lows (simplified)
        swing_high = high.rolling(5, center=True).max().dropna()
        swing_low = low.rolling(5, center=True).min().dropna()

        # Key levels near current price
        resistance_levels = sorted(set([
            round(r1, 2), round(r2, 2),
            round(high.tail(20).max(), 2),
        ]))
        support_levels = sorted(set([
            round(s1, 2), round(s2, 2),
            round(low.tail(20).min(), 2),
        ]), reverse=True)

        return {
            "pivot": round(pivot, 2),
            "resistance_1": round(r1, 2),
            "resistance_2": round(r2, 2),
            "support_1": round(s1, 2),
            "support_2": round(s2, 2),
            "twenty_day_high": round(high.tail(20).max(), 2),
            "twenty_day_low": round(low.tail(20).min(), 2),
        }

    # ── IV Data ───────────────────────────────────────────────────────

    def _get_iv_data(self, stock, expirations: List[str], current_price: float) -> Dict:
        """Extract implied volatility data from the options chain."""
        if not expirations:
            return {"avg_iv": None, "iv_rank": None, "has_affordable": False}

        try:
            exp = expirations[0]
            chain = stock.option_chain(exp)
            calls = chain.calls
            puts = chain.puts

            ntm_calls = calls[
                (calls["strike"] >= current_price * 0.90) &
                (calls["strike"] <= current_price * 1.10)
            ]
            ntm_puts = puts[
                (puts["strike"] >= current_price * 0.90) &
                (puts["strike"] <= current_price * 1.10)
            ]

            all_iv = pd.concat([
                ntm_calls["impliedVolatility"] if "impliedVolatility" in ntm_calls.columns else pd.Series(dtype=float),
                ntm_puts["impliedVolatility"] if "impliedVolatility" in ntm_puts.columns else pd.Series(dtype=float),
            ])
            avg_iv = round(all_iv.mean(), 4) if not all_iv.empty else None

            affordable_calls = calls[calls["lastPrice"] <= self.max_premium]
            affordable_puts = puts[puts["lastPrice"] <= self.max_premium]
            has_affordable = len(affordable_calls) > 0 or len(affordable_puts) > 0

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

            iv_rank = None
            if avg_iv:
                full_hist = stock.history(period="3mo")
                if not full_hist.empty:
                    ret = full_hist["Close"].pct_change().dropna()
                    hv_90d = ret.std() * np.sqrt(252)
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

    # ── Scoring & Ranking ─────────────────────────────────────────────

    def run_scan(self, max_workers: int = 8) -> List[Dict]:
        """Run the full screener across the watchlist."""
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

        for r in results:
            r["composite_score"] = self._calculate_composite_score(r)

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results

    def _calculate_composite_score(self, data: Dict) -> float:
        """
        Calculate a composite opportunity score (0-100) based on:
        - Volatility, Volume, Momentum, IV rank, ATR%
        - 200 EMA test bonus
        - Squeeze potential bonus
        - Unusual options activity bonus
        """
        score = 0.0

        # Volatility component (0-15 pts)
        hv = data.get("hist_volatility_20d", 0)
        score += min(hv * 30, 15)

        # Volume component (0-10 pts)
        vol = data.get("avg_volume", 0)
        if vol > 50_000_000:
            score += 10
        elif vol > 10_000_000:
            score += 8
        elif vol > 5_000_000:
            score += 5
        elif vol > 1_000_000:
            score += 3

        # Momentum magnitude (0-10 pts)
        mom_1d = abs(data.get("momentum_1d", 0))
        mom_5d = abs(data.get("momentum_5d", 0))
        score += min(mom_1d * 150, 5)
        score += min(mom_5d * 30, 5)

        # IV rank component (0-10 pts)
        iv_rank = data.get("iv_rank")
        if iv_rank and iv_rank > 1.5:
            score += 10
        elif iv_rank and iv_rank > 1.2:
            score += 7
        elif iv_rank and iv_rank > 1.0:
            score += 3

        # ATR% component (0-10 pts)
        atr_pct = data.get("atr_pct", 0)
        score += min(atr_pct * 200, 10)

        # ── 200 EMA Test Bonus (0-20 pts) ──
        ema_data = data.get("ema_200", {})
        if not ema_data.get("insufficient_data"):
            if ema_data.get("testing"):
                score += 12  # Currently testing the 200 EMA
            touch_count = ema_data.get("touch_count_30d", 0)
            if touch_count >= 1:
                score += min(touch_count * 3, 8)  # More touches = more significant level
            pattern = ema_data.get("bounce_pattern")
            if pattern in ("BULLISH_BOUNCE", "BEARISH_REJECTION"):
                score += 5  # Clear directional signal after EMA test

        # ── Squeeze Bonus (0-25 pts) ──
        squeeze = data.get("squeeze", {})
        squeeze_score = squeeze.get("squeeze_score", 0)
        score += squeeze_score * 0.25  # Scale 0-100 squeeze score to 0-25

        # ── Unusual Options Activity Bonus (0-10 pts) ──
        uoa = data.get("unusual_options_activity", {})
        if uoa.get("unusual"):
            signal_count = uoa.get("signal_count", 0)
            score += min(signal_count * 3, 10)

        # ── Earnings Catalyst (0-5 pts) ──
        if data.get("earnings_within_7d"):
            score += 5

        return round(score, 2)

    def get_top_candidates(self, n: int = 10) -> List[Dict]:
        """Get the top N candidates from the scan."""
        all_results = self.run_scan()
        return all_results[:n]
