"""
Gann Sentinel Trader - Grok Scanner
Forward-looking sentiment and news signal extraction via xAI Grok API.

Version: 2.2.0 (Live Search API Fix)
Last Updated: January 2026

Change Log:
- 2.2.0: Fixed xAI API - use search_parameters instead of tools for Live Search
         Note: Live Search API deprecated Jan 12, 2026 - migration to Agent Tools API pending
"""

import os
import uuid
import hashlib
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import httpx

# Import our temporal framework
from scanners.temporal import (
    TemporalContext,
    TemporalQueryBuilder,
    TimeHorizon,
    SignalRelevance,
    get_temporal_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class GrokModel(Enum):
    """Available Grok models."""
    GROK_FAST = "grok-3-fast-beta"
    GROK_REASONING = "grok-3-mini-fast-beta"


class SignalCategory(Enum):
    """Categories of signals Grok can generate."""
    SENTIMENT = "sentiment"
    NEWS = "news"
    NARRATIVE_SHIFT = "narrative_shift"
    EVENT = "event"


# Sector mappings for asset scope
TICKER_TO_SECTOR = {
    "NVDA": "TECH", "AMD": "TECH", "SMCI": "TECH", "AAPL": "TECH",
    "MSFT": "TECH", "GOOGL": "TECH", "META": "TECH", "AMZN": "TECH",
    "JPM": "FINANCIALS", "GS": "FINANCIALS", "MS": "FINANCIALS",
    "XOM": "ENERGY", "CVX": "ENERGY", "OXY": "ENERGY",
    "RKLB": "AEROSPACE", "LMT": "AEROSPACE", "RTX": "AEROSPACE",
    "COIN": "CRYPTO", "MSTR": "CRYPTO",
    "TSLA": "AUTO",
    "SPY": "INDEX", "QQQ": "INDEX", "IWM": "INDEX",
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class GrokSignal:
    """Signal extracted from Grok conforming to Grok Spec v1.1.0."""
    signal_id: str
    dedup_hash: str
    category: str
    source_type: str
    asset_scope: Dict[str, List[str]]
    summary: str
    raw_value: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    confidence_factors: Dict[str, float]
    directional_bias: str
    time_horizon: str
    novelty: str
    staleness_policy: Dict[str, Any]
    uncertainties: List[str]
    timestamp_utc: str
    forward_horizon: Optional[str] = None
    catalyst_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "signal_id": self.signal_id,
            "dedup_hash": self.dedup_hash,
            "signal_type": self.category,
            "category": self.category,
            "source_type": self.source_type,
            "source": self.source_type,
            "asset_scope": self.asset_scope,
            "summary": self.summary,
            "raw_value": self.raw_value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "confidence_factors": self.confidence_factors,
            "directional_bias": self.directional_bias,
            "time_horizon": self.time_horizon,
            "novelty": self.novelty,
            "staleness_policy": self.staleness_policy,
            "staleness_seconds": self.staleness_policy.get("max_age_seconds", 3600),
            "uncertainties": self.uncertainties,
            "timestamp_utc": self.timestamp_utc,
            "forward_horizon": self.forward_horizon,
            "catalyst_date": self.catalyst_date,
        }


# =============================================================================
# GROK SCANNER
# =============================================================================

class GrokScanner:
    """
    Scanner for sentiment and news signals via xAI Grok API.
    
    Enhanced with better error handling and diagnostics.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Grok scanner."""
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        
        if not self.api_key:
            logger.warning("XAI_API_KEY not set - Grok scanning disabled")
        else:
            logger.info(f"XAI_API_KEY configured (length: {len(self.api_key)})")
        
        self.base_url = "https://api.x.ai/v1"
        self.model = GrokModel.GROK_FAST.value
        
        # Initialize temporal framework
        self.temporal_context = get_temporal_context()
        self.query_builder = TemporalQueryBuilder(self.temporal_context)
        
        # Cache for deduplication
        self._seen_signals: Dict[str, datetime] = {}
        
        # ERROR TRACKING for diagnostics
        self.last_error: Optional[str] = None
        self.last_raw_response: Optional[str] = None
        
        logger.info("GrokScanner initialized")
    
    @property
    def is_configured(self) -> bool:
        """Check if scanner is properly configured."""
        return bool(self.api_key)
    
    # =========================================================================
    # SIMPLIFIED PROMPTS (more likely to get valid JSON)
    # =========================================================================
    
    def _build_simple_sentiment_prompt(self, tickers: List[str]) -> str:
        """Build a simpler prompt that's more likely to return valid JSON."""
        ticker_str = ", ".join(tickers)
        
        return f"""Analyze market sentiment for these stocks: {ticker_str}

Respond ONLY with valid JSON in this exact format:
{{
    "tickers": [
        {{
            "symbol": "TICKER",
            "sentiment": "bullish" or "bearish" or "neutral",
            "score": 0.5,
            "summary": "One sentence about outlook"
        }}
    ]
}}

Include one entry per ticker. No other text, just the JSON."""

    def _build_simple_outlook_prompt(self) -> str:
        """Build a simpler market outlook prompt."""
        return """Analyze current US stock market outlook.

Respond ONLY with valid JSON:
{
    "outlook": "bullish" or "bearish" or "neutral",
    "confidence": 0.5,
    "summary": "Brief market outlook",
    "key_factors": ["factor 1", "factor 2"]
}

No other text, just the JSON."""

    # =========================================================================
    # API CALLS WITH ENHANCED ERROR HANDLING
    # =========================================================================
    
    async def _call_grok(
        self,
        system_prompt: str,
        user_message: str,
        use_search: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the Grok API with enhanced error handling and logging.
        """
        self.last_error = None
        self.last_raw_response = None
        
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
        }
        
        if use_search:
            # xAI Live Search API uses search_parameters, not tools
            # mode: "auto" lets Grok decide when to search, "on" forces search
            # Sources default to web, news, and x when not specified
            request_body["search_parameters"] = {
                "mode": "auto",
                "return_citations": True,
            }
        
        try:
            logger.info(f"Calling Grok API with model {self.model}...")
            logger.debug(f"Request body keys: {list(request_body.keys())}")
            if use_search:
                logger.info(f"Search enabled with parameters: {request_body.get('search_parameters', {})}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                
                logger.info(f"Grok API response status: {response.status_code}")
                
                if response.status_code != 200:
                    self.last_error = f"API error {response.status_code}: {response.text[:200]}"
                    logger.error(self.last_error)
                    return None
                
                data = response.json()
                
                # Extract content from response
                if "choices" not in data or not data["choices"]:
                    self.last_error = "No choices in API response"
                    logger.error(self.last_error)
                    return None
                
                content = data["choices"][0].get("message", {}).get("content", "")
                
                if not content:
                    self.last_error = "Empty content in API response"
                    logger.error(self.last_error)
                    return None
                
                # Store raw response for debugging
                self.last_raw_response = content[:500]
                logger.info(f"Grok raw response (first 200 chars): {content[:200]}")
                
                # Try to parse JSON
                parsed = self._extract_json(content)
                
                if parsed:
                    logger.info(f"Successfully parsed JSON with keys: {list(parsed.keys())}")
                    return parsed
                else:
                    self.last_error = f"Could not parse JSON from response"
                    logger.warning(self.last_error)
                    # Return raw content wrapped in dict
                    return {"_raw": content, "_parse_failed": True}
                    
        except httpx.TimeoutException:
            self.last_error = "API timeout (60s)"
            logger.error(self.last_error)
            return None
        except httpx.ConnectError as e:
            self.last_error = f"Connection error: {str(e)[:100]}"
            logger.error(self.last_error)
            return None
        except Exception as e:
            self.last_error = f"API error: {str(e)[:100]}"
            logger.error(self.last_error)
            return None
    
    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Try multiple methods to extract JSON from response.
        """
        # Method 1: Try direct parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from markdown code block
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass
        
        if "```" in content:
            try:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass
        
        # Method 3: Find JSON object with regex
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Method 4: Find JSON array
        array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        matches = re.findall(array_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                arr = json.loads(match)
                return {"items": arr}
            except json.JSONDecodeError:
                continue
        
        return None
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _generate_dedup_hash(self, source_type: str, primary_asset: str, summary: str) -> str:
        """Generate deduplication hash."""
        normalized = f"{source_type}:{primary_asset}:{summary.lower().strip()[:100]}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _create_fallback_signal(
        self,
        source_type: str,
        summary: str,
        tickers: List[str] = None
    ) -> GrokSignal:
        """Create a fallback signal when JSON parsing fails but we got content."""
        now = datetime.now(timezone.utc)
        signal_id = str(uuid.uuid4())
        
        primary_ticker = tickers[0] if tickers else "MARKET"
        dedup_hash = self._generate_dedup_hash(source_type, primary_ticker, summary)
        
        return GrokSignal(
            signal_id=signal_id,
            dedup_hash=dedup_hash,
            category="sentiment",
            source_type=source_type,
            asset_scope={
                "tickers": tickers or [],
                "sectors": [],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },
            summary=summary[:200],
            raw_value={
                "type": "index",
                "value": 0.5,
                "unit": "sentiment",
                "prior_value": None,
                "change": None,
                "change_period": None,
            },
            evidence=[{
                "source": source_type,
                "source_tier": "social",
                "excerpt": "Grok analysis (unparsed format)",
                "timestamp_utc": now.isoformat(),
            }],
            confidence=0.30,  # Lower confidence for fallback
            confidence_factors={
                "source_base": 0.30,
                "recency_factor": 1.0,
                "corroboration_factor": 1.0,
            },
            directional_bias="unclear",
            time_horizon="weeks",
            novelty="new",
            staleness_policy={
                "max_age_seconds": 3600,
                "stale_after_utc": (now + timedelta(hours=1)).isoformat(),
            },
            uncertainties=["Response format was not parseable as expected JSON"],
            timestamp_utc=now.isoformat(),
            forward_horizon="short-term",
        )
    
    def _parse_sentiment_to_signals(
        self,
        response: Dict[str, Any],
        tickers: List[str],
    ) -> List[GrokSignal]:
        """Parse Grok sentiment response into signals."""
        signals = []
        now = datetime.now(timezone.utc)
        
        # Check if this is a failed parse
        if response.get("_parse_failed"):
            raw_content = response.get("_raw", "")
            if raw_content:
                # Create a single fallback signal with the raw content
                logger.info("Creating fallback signal from raw content")
                summary = raw_content[:150].replace("\n", " ")
                signals.append(self._create_fallback_signal("grok_x", summary, tickers))
            return signals
        
        # Try to get tickers from response
        tickers_data = response.get("tickers", [])
        
        if not tickers_data:
            logger.warning(f"No 'tickers' key in response. Keys found: {list(response.keys())}")
            # Try alternative structures
            if "items" in response:
                tickers_data = response["items"]
            elif isinstance(response, list):
                tickers_data = response
        
        for ticker_data in tickers_data:
            try:
                symbol = ticker_data.get("symbol") or ticker_data.get("ticker", "UNKNOWN")
                sentiment = ticker_data.get("sentiment") or ticker_data.get("forward_sentiment", "neutral")
                score = float(ticker_data.get("score") or ticker_data.get("sentiment_score", 0.5))
                summary_text = ticker_data.get("summary") or ticker_data.get("key_expectations", ["No details"])[0]
                
                signal_id = str(uuid.uuid4())
                full_summary = f"{symbol}: {sentiment}. {summary_text}"
                dedup_hash = self._generate_dedup_hash("grok_x", symbol, full_summary)
                
                if dedup_hash in self._seen_signals:
                    continue
                self._seen_signals[dedup_hash] = now
                
                sector = TICKER_TO_SECTOR.get(symbol, "")
                
                if sentiment.lower() in ["bullish", "positive"]:
                    directional_bias = "positive"
                elif sentiment.lower() in ["bearish", "negative"]:
                    directional_bias = "negative"
                elif sentiment.lower() == "mixed":
                    directional_bias = "mixed"
                else:
                    directional_bias = "unclear"
                
                signal = GrokSignal(
                    signal_id=signal_id,
                    dedup_hash=dedup_hash,
                    category="sentiment",
                    source_type="grok_x",
                    asset_scope={
                        "tickers": [symbol],
                        "sectors": [sector] if sector else [],
                        "macro_regions": ["US"],
                        "asset_classes": ["EQUITY"],
                    },
                    summary=full_summary[:200],
                    raw_value={
                        "type": "index",
                        "value": score,
                        "unit": "sentiment_score",
                        "prior_value": None,
                        "change": None,
                        "change_period": None,
                    },
                    evidence=[{
                        "source": "grok_x_search",
                        "source_tier": "social",
                        "excerpt": summary_text[:100],
                        "timestamp_utc": now.isoformat(),
                    }],
                    confidence=min(0.40 * (1 + score), 0.80),
                    confidence_factors={
                        "source_base": 0.40,
                        "recency_factor": 1.0,
                        "corroboration_factor": 1.0,
                    },
                    directional_bias=directional_bias,
                    time_horizon="weeks",
                    novelty="new",
                    staleness_policy={
                        "max_age_seconds": 3600,
                        "stale_after_utc": (now + timedelta(hours=1)).isoformat(),
                    },
                    uncertainties=[],
                    timestamp_utc=now.isoformat(),
                    forward_horizon="short-term",
                )
                
                signals.append(signal)
                logger.info(f"Created signal for {symbol}: {sentiment}")
                
            except Exception as e:
                logger.error(f"Error parsing ticker data: {e}")
                continue
        
        return signals
    
    def _parse_outlook_to_signals(self, response: Dict[str, Any]) -> List[GrokSignal]:
        """Parse market outlook response into signals."""
        signals = []
        now = datetime.now(timezone.utc)
        
        # Check if this is a failed parse
        if response.get("_parse_failed"):
            raw_content = response.get("_raw", "")
            if raw_content:
                summary = raw_content[:150].replace("\n", " ")
                signals.append(self._create_fallback_signal("grok_web", f"Market: {summary}"))
            return signals
        
        try:
            outlook = response.get("outlook") or response.get("overall_outlook", "neutral")
            confidence = float(response.get("confidence", 0.5))
            summary = response.get("summary") or response.get("horizon_end_of_month", "Market outlook analysis")
            key_factors = response.get("key_factors") or response.get("key_themes", [])
            
            signal_id = str(uuid.uuid4())
            full_summary = f"Market outlook: {outlook}. {summary}"
            dedup_hash = self._generate_dedup_hash("grok_web", "MARKET", full_summary)
            
            if outlook.lower() in ["bullish", "positive"]:
                directional_bias = "positive"
            elif outlook.lower() in ["bearish", "negative"]:
                directional_bias = "negative"
            else:
                directional_bias = "mixed"
            
            signal = GrokSignal(
                signal_id=signal_id,
                dedup_hash=dedup_hash,
                category="sentiment",
                source_type="grok_web",
                asset_scope={
                    "tickers": [],
                    "sectors": [],
                    "macro_regions": ["US"],
                    "asset_classes": ["EQUITY"],
                },
                summary=full_summary[:200],
                raw_value={
                    "type": "index",
                    "value": confidence,
                    "unit": "outlook_confidence",
                    "prior_value": None,
                    "change": None,
                    "change_period": None,
                },
                evidence=[{
                    "source": "grok_web_search",
                    "source_tier": "tier2",
                    "excerpt": "; ".join(key_factors[:3]) if key_factors else summary[:100],
                    "timestamp_utc": now.isoformat(),
                }],
                confidence=min(0.55 * (1 + confidence), 0.80),
                confidence_factors={
                    "source_base": 0.55,
                    "recency_factor": 1.0,
                    "corroboration_factor": 1.0,
                },
                directional_bias=directional_bias,
                time_horizon="weeks",
                novelty="new",
                staleness_policy={
                    "max_age_seconds": 14400,  # 4 hours
                    "stale_after_utc": (now + timedelta(hours=4)).isoformat(),
                },
                uncertainties=response.get("risks_to_watch", [])[:3],
                timestamp_utc=now.isoformat(),
                forward_horizon="end of quarter",
            )
            
            signals.append(signal)
            logger.info(f"Created market outlook signal: {outlook}")
            
        except Exception as e:
            logger.error(f"Error parsing market outlook: {e}")
        
        return signals
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    async def scan_sentiment(
        self, 
        tickers: List[str],
        horizon: TimeHorizon = TimeHorizon.SHORT_TERM,
    ) -> List[GrokSignal]:
        """
        Scan for sentiment on specific tickers.
        """
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return []
        
        logger.info(f"Scanning sentiment for {tickers}")
        
        # Use simpler prompt for better JSON parsing
        system_prompt = self._build_simple_sentiment_prompt(tickers)
        user_message = f"Analyze sentiment for: {', '.join(tickers)}"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            logger.warning(f"No response from Grok. Last error: {self.last_error}")
            return []
        
        signals = self._parse_sentiment_to_signals(response, tickers)
        logger.info(f"Generated {len(signals)} sentiment signals")
        return signals
    
    async def scan_market_overview(self) -> List[GrokSignal]:
        """
        Scan for overall market outlook.
        """
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return []
        
        logger.info("Scanning market overview")
        
        system_prompt = self._build_simple_outlook_prompt()
        user_message = "What is the current US stock market outlook?"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            logger.warning(f"No response from Grok. Last error: {self.last_error}")
            return []
        
        signals = self._parse_outlook_to_signals(response)
        logger.info(f"Generated {len(signals)} market overview signals")
        return signals
    
    async def scan_catalysts(self, ticker: str) -> List[GrokSignal]:
        """
        Scan for upcoming catalysts for a specific ticker.
        """
        if not self.is_configured:
            return []
        
        logger.info(f"Scanning catalysts for {ticker}")
        
        system_prompt = f"""Find upcoming events and catalysts for {ticker}.
Respond ONLY with valid JSON:
{{
    "catalysts": [
        {{"event": "...", "date": "...", "impact": "high/medium/low"}}
    ]
}}"""
        
        user_message = f"What are upcoming catalysts for {ticker}?"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            return []
        
        # Parse catalysts
        signals = []
        now = datetime.now(timezone.utc)
        
        catalysts = response.get("catalysts", [])
        for catalyst in catalysts[:5]:
            try:
                event = catalyst.get("event", "Unknown event")
                date = catalyst.get("date", "TBD")
                impact = catalyst.get("impact", "medium")
                
                signal_id = str(uuid.uuid4())
                summary = f"{ticker}: {event} ({date})"
                dedup_hash = self._generate_dedup_hash("grok_web", ticker, summary)
                
                if dedup_hash in self._seen_signals:
                    continue
                self._seen_signals[dedup_hash] = now
                
                signal = GrokSignal(
                    signal_id=signal_id,
                    dedup_hash=dedup_hash,
                    category="event",
                    source_type="grok_web",
                    asset_scope={
                        "tickers": [ticker],
                        "sectors": [TICKER_TO_SECTOR.get(ticker, "")],
                        "macro_regions": ["US"],
                        "asset_classes": ["EQUITY"],
                    },
                    summary=summary[:200],
                    raw_value={
                        "type": "null",
                        "value": None,
                        "unit": None,
                        "prior_value": None,
                        "change": None,
                        "change_period": None,
                    },
                    evidence=[{
                        "source": "grok_web_search",
                        "source_tier": "tier2",
                        "excerpt": event,
                        "timestamp_utc": now.isoformat(),
                    }],
                    confidence=0.60 if impact == "high" else 0.45,
                    confidence_factors={
                        "source_base": 0.55,
                        "recency_factor": 1.0,
                        "corroboration_factor": 1.0,
                    },
                    directional_bias="unclear",
                    time_horizon="weeks",
                    novelty="new",
                    staleness_policy={
                        "max_age_seconds": 86400,
                        "stale_after_utc": (now + timedelta(hours=24)).isoformat(),
                    },
                    uncertainties=[f"Date: {date}"],
                    timestamp_utc=now.isoformat(),
                    forward_horizon="weeks",
                    catalyst_date=date,
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error parsing catalyst: {e}")
                continue
        
        return signals


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def scan_grok_sentiment(tickers: List[str]) -> List[Dict[str, Any]]:
    """Convenience function to run a Grok sentiment scan."""
    scanner = GrokScanner()
    signals = await scanner.scan_sentiment(tickers)
    return [s.to_dict() for s in signals]


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def test():
        scanner = GrokScanner()
        
        print("\n" + "="*60)
        print("GROK SCANNER TEST")
        print("="*60)
        
        if scanner.is_configured:
            print("\nTesting sentiment scan...")
            signals = await scanner.scan_sentiment(["NVDA", "AAPL"])
            print(f"Got {len(signals)} sentiment signals")
            for sig in signals:
                print(f"  - {sig.summary[:60]}...")
            
            if scanner.last_error:
                print(f"\nLast error: {scanner.last_error}")
            if scanner.last_raw_response:
                print(f"\nRaw response preview: {scanner.last_raw_response[:200]}...")
        else:
            print("\nXAI_API_KEY not set - skipping live test")
    
    asyncio.run(test())
