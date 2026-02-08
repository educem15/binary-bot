"""
Daily Options Report Generator
Produces a ranked morning watchlist with actionable trade candidates.
Combines screening + deep analysis into a single daily briefing.
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

from app.options.screener import OptionsScreener
from app.options.analyzer import OptionsAnalyzer

logger = logging.getLogger(__name__)


class DailyOptionsReport:
    """Generates the daily morning options research report."""

    def __init__(self, budget: float = 500.0, watchlist: Optional[List[str]] = None):
        self.budget = budget
        self.screener = OptionsScreener(budget=budget, watchlist=watchlist)
        self.analyzer = OptionsAnalyzer(budget=budget)

    def generate_report(
        self,
        top_n: int = 10,
        deep_analysis_n: int = 5,
        expiration: Optional[str] = None,
    ) -> Dict:
        """
        Generate the full daily report.

        1. Screen entire watchlist for top opportunities
        2. Deep-analyze the top N candidates
        3. Rank and present with actionable strategies

        Args:
            top_n: Number of top screener results to include
            deep_analysis_n: Number of those to deep-analyze with Greeks/strategies
            expiration: Optional specific expiration date to target
        """
        report_time = datetime.now().isoformat()

        # Step 1: Screen
        logger.info("Running options screener...")
        screened = self.screener.get_top_candidates(n=top_n)

        # Step 2: Deep-analyze top candidates
        logger.info(f"Deep-analyzing top {deep_analysis_n} candidates...")
        deep_analyses = []
        for candidate in screened[:deep_analysis_n]:
            ticker = candidate["ticker"]
            try:
                analysis = self.analyzer.analyze_ticker(ticker, expiration=expiration)
                deep_analyses.append({
                    "screening_data": candidate,
                    "analysis": analysis,
                })
            except Exception as e:
                logger.warning(f"Could not deep-analyze {ticker}: {e}")
                deep_analyses.append({
                    "screening_data": candidate,
                    "analysis": None,
                    "error": str(e),
                })

        # Step 3: Build the ranked watchlist
        watchlist = self._build_ranked_watchlist(deep_analyses)

        # Step 4: Build summary
        summary = self._build_summary(screened, deep_analyses, watchlist)

        return {
            "report_time": report_time,
            "budget": self.budget,
            "summary": summary,
            "watchlist": watchlist,
            "screener_results": screened,
            "detailed_analyses": deep_analyses,
        }

    def _build_ranked_watchlist(self, analyses: List[Dict]) -> List[Dict]:
        """Build a simplified ranked watchlist for quick decision-making."""
        watchlist = []

        for item in analyses:
            screening = item["screening_data"]
            analysis = item.get("analysis")

            entry = {
                "rank": 0,
                "ticker": screening["ticker"],
                "price": screening.get("current_price", 0),
                "composite_score": screening.get("composite_score", 0),
                "beta": screening.get("beta"),
                "atr_pct": screening.get("atr_pct"),
                "momentum_1d": screening.get("momentum_1d"),
                "implied_volatility": screening.get("implied_volatility"),
                "squeeze_score": screening.get("squeeze", {}).get("squeeze_score", 0),
                "ema_200_testing": screening.get("ema_200", {}).get("testing", False),
                "unusual_activity": screening.get("unusual_options_activity", {}).get("unusual", False),
            }

            if analysis:
                entry["bias"] = analysis["technical_bias"]["direction"]
                entry["bias_confidence"] = analysis["technical_bias"]["confidence"]
                entry["rsi"] = analysis["technical_bias"]["rsi"]
                entry["num_strategies"] = len(analysis.get("strategies", []))
                entry["top_strategy"] = (
                    analysis["strategies"][0]["name"]
                    if analysis.get("strategies")
                    else "No strategy found"
                )
                entry["top_strategy_cost"] = (
                    analysis["strategies"][0]["max_loss"]
                    if analysis.get("strategies")
                    else None
                )
                entry["top_strategy_risk"] = (
                    analysis["strategies"][0]["risk_level"]
                    if analysis.get("strategies")
                    else None
                )
                # Boost score if strong bias alignment
                if analysis["technical_bias"]["confidence"] >= 0.75:
                    entry["composite_score"] *= 1.15
            else:
                entry["bias"] = "UNKNOWN"
                entry["bias_confidence"] = 0
                entry["num_strategies"] = 0
                entry["top_strategy"] = "Analysis failed"

            watchlist.append(entry)

        # Sort by composite score
        watchlist.sort(key=lambda x: x["composite_score"], reverse=True)

        # Assign ranks
        for i, item in enumerate(watchlist):
            item["rank"] = i + 1
            item["composite_score"] = round(item["composite_score"], 2)

        return watchlist

    def _build_summary(
        self, screened: List[Dict], analyses: List[Dict], watchlist: List[Dict]
    ) -> Dict:
        """Build a high-level summary of the day's opportunities."""
        bullish = [w for w in watchlist if w.get("bias") == "BULLISH"]
        bearish = [w for w in watchlist if w.get("bias") == "BEARISH"]
        neutral = [w for w in watchlist if w.get("bias") == "NEUTRAL"]

        avg_iv = None
        ivs = [s.get("implied_volatility") for s in screened if s.get("implied_volatility")]
        if ivs:
            avg_iv = round(sum(ivs) / len(ivs), 4)

        strategies_available = sum(w.get("num_strategies", 0) for w in watchlist)

        # Count squeeze and EMA setups from screener data
        squeeze_setups = sum(1 for s in screened if s.get("squeeze", {}).get("squeeze_score", 0) > 30)
        ema_testing = sum(1 for s in screened if s.get("ema_200", {}).get("testing", False))
        unusual_activity_count = sum(1 for s in screened if s.get("unusual_options_activity", {}).get("unusual", False))

        return {
            "total_tickers_scanned": len(self.screener.watchlist),
            "qualifying_tickers": len(screened),
            "deeply_analyzed": len([a for a in analyses if a.get("analysis")]),
            "bullish_setups": len(bullish),
            "bearish_setups": len(bearish),
            "neutral_setups": len(neutral),
            "average_implied_volatility": avg_iv,
            "total_strategies_found": strategies_available,
            "squeeze_setups": squeeze_setups,
            "ema_200_testing": ema_testing,
            "unusual_activity_tickers": unusual_activity_count,
            "market_mood": (
                "BULLISH" if len(bullish) > len(bearish) * 1.5
                else "BEARISH" if len(bearish) > len(bullish) * 1.5
                else "MIXED"
            ),
            "top_pick": watchlist[0] if watchlist else None,
        }

    def generate_text_report(self, top_n: int = 10, deep_analysis_n: int = 5) -> str:
        """Generate a human-readable text version of the report."""
        data = self.generate_report(top_n=top_n, deep_analysis_n=deep_analysis_n)
        lines = []

        lines.append("=" * 70)
        lines.append(f"  DAILY OPTIONS RESEARCH REPORT - {data['report_time'][:10]}")
        lines.append(f"  Budget: ${data['budget']:.2f} per trade")
        lines.append("=" * 70)

        # Summary
        s = data["summary"]
        lines.append("")
        lines.append("MARKET OVERVIEW")
        lines.append("-" * 40)
        lines.append(f"  Tickers scanned:     {s['total_tickers_scanned']}")
        lines.append(f"  Qualifying:          {s['qualifying_tickers']}")
        lines.append(f"  Deep analyzed:       {s['deeply_analyzed']}")
        lines.append(f"  Market mood:         {s['market_mood']}")
        lines.append(f"  Avg IV:              {s['average_implied_volatility']}")
        lines.append(f"  Bullish setups:      {s['bullish_setups']}")
        lines.append(f"  Bearish setups:      {s['bearish_setups']}")
        lines.append(f"  Total strategies:    {s['total_strategies_found']}")
        lines.append("")

        # Watchlist
        lines.append("TOP PICKS (Ranked)")
        lines.append("-" * 70)
        lines.append(
            f"  {'#':<3} {'Ticker':<8} {'Price':>8} {'Score':>7} "
            f"{'Bias':<8} {'RSI':>6} {'ATR%':>7} {'Strategy':<28}"
        )
        lines.append("  " + "-" * 68)

        for w in data["watchlist"]:
            rsi_str = f"{w.get('rsi', 'N/A')}" if w.get('rsi') else "N/A"
            atr_str = f"{w.get('atr_pct', 0) * 100:.1f}%" if w.get('atr_pct') else "N/A"
            lines.append(
                f"  {w['rank']:<3} {w['ticker']:<8} ${w['price']:>7.2f} "
                f"{w['composite_score']:>6.1f} {w.get('bias', 'N/A'):<8} "
                f"{rsi_str:>6} {atr_str:>7} {w.get('top_strategy', 'N/A'):<28}"
            )

        # Detailed strategies for top picks
        lines.append("")
        lines.append("DETAILED STRATEGIES")
        lines.append("=" * 70)

        for item in data["detailed_analyses"]:
            analysis = item.get("analysis")
            if not analysis:
                continue

            ticker = analysis["ticker"]
            lines.append("")
            lines.append(f"--- {ticker} @ ${analysis['current_price']:.2f} ---")
            lines.append(
                f"  Bias: {analysis['technical_bias']['direction']} "
                f"(confidence: {analysis['technical_bias']['confidence']:.0%})"
            )
            lines.append(f"  HV: {analysis['historical_volatility']:.1%}  |  "
                         f"Expiration: {analysis['expiration']}  |  "
                         f"DTE: {analysis['days_to_expiration']}")

            if analysis.get("top_calls"):
                best_call = analysis["top_calls"][0]
                lines.append(
                    f"  Best Call: ${best_call['strike']} @ ${best_call['premium']:.2f} "
                    f"(cost: ${best_call['total_cost']:.2f})  "
                    f"Delta: {best_call['delta']}  Theta: {best_call['theta']}  "
                    f"PoP: {best_call['probability_of_profit']:.1%}"
                )

            if analysis.get("top_puts"):
                best_put = analysis["top_puts"][0]
                lines.append(
                    f"  Best Put:  ${best_put['strike']} @ ${best_put['premium']:.2f} "
                    f"(cost: ${best_put['total_cost']:.2f})  "
                    f"Delta: {best_put['delta']}  Theta: {best_put['theta']}  "
                    f"PoP: {best_put['probability_of_profit']:.1%}"
                )

            for strat in analysis.get("strategies", []):
                lines.append(f"")
                lines.append(f"  >> STRATEGY: {strat['name']}")
                lines.append(f"     Risk: {strat['risk_level']}  |  "
                             f"Max Loss: ${strat['max_loss']}  |  "
                             f"Max Gain: {strat['max_gain']}")
                lines.append(f"     Breakeven: {strat['breakeven']}")
                lines.append(f"     Rationale: {strat['rationale']}")
                lines.append(f"     Exit Plan: {strat['ideal_exit']}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("DISCLAIMER: This is research data only. All trading decisions are yours.")
        lines.append("Past performance does not guarantee future results.")
        lines.append("=" * 70)

        return "\n".join(lines)
