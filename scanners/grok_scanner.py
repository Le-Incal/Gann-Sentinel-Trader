"""
Gann Sentinel Trader - Grok Scanner
Uses xAI's Grok API for sentiment analysis and news gathering.
"""

import logging
import httpx
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from config import Config
from models.signals import Signal, SignalType, SignalSource, DirectionalBias, TimeHorizon, AssetScope, RawValue, Evidence

logger = logging.getLogger(__name__)


class GrokScanner:
    """
    Scans X/Twitter and web for market sentiment and news using Grok API.
    """
    
    def __init__(self):
        """Initialize Grok scanner."""
        self.api_key = Config.XAI_API_KEY
        self.base_url = Config.XAI_BASE_URL
        self.model = Config.XAI_MODEL
        
        if not self.api_key:
            logger.warning("XAI_API_KEY not configured - Grok scanner disabled")
    
    async def _call_grok(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Make a call to the Grok API."""
        if not self.api_key:
            raise ValueError("XAI_API_KEY not configured")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        if tools:
            payload["tools"] = tools
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def scan_sentiment(
        self,
        tickers: List[str],
        lookback_hours: int = 24
    ) -> List[Signal]:
        """
        Scan X/Twitter for sentiment on given tickers.
        
        Args:
            tickers: List of stock tickers to scan
            lookback_hours: How far back to look
            
        Returns:
            List of Signal objects
        """
        if not self.api_key:
            logger.warning("Grok scanner not configured - returning empty signals")
            return []
        
        signals = []
        
        for ticker in tickers:
            try:
                signal = await self._scan_ticker_sentiment(ticker, lookback_hours)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error scanning sentiment for {ticker}: {e}")
        
        return signals
    
    async def _scan_ticker_sentiment(
        self,
        ticker: str,
        lookback_hours: int
    ) -> Optional[Signal]:
        """Scan sentiment for a single ticker."""
        
        system_prompt = """You are a market sentiment analyst. Analyze X/Twitter sentiment for the given stock ticker.

You have access to x_search to search X/Twitter posts. Use it to find recent discussions about the ticker.

Provide a structured analysis including:
1. Overall sentiment (bullish, bearish, neutral, mixed)
2. Sentiment score (-1.0 to 1.0)
3. Key narratives being discussed
4. Notable posts from influential accounts
5. Volume/activity level compared to normal
6. Any emerging themes or catalysts mentioned

Be factual and specific. Cite actual posts when possible."""

        user_prompt = f"""Analyze the current X/Twitter sentiment for ${ticker}.

Look for:
- Recent posts mentioning ${ticker} or the company
- Sentiment of retail traders
- Any breaking news or catalysts
- Influential accounts discussing this stock
- Unusual activity or volume in discussions

Provide a sentiment score from -1.0 (extremely bearish) to 1.0 (extremely bullish)."""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "x_search",
                    "description": "Search X/Twitter for posts about a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        try:
            response = await self._call_grok(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=tools
            )
            
            # Parse response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract sentiment from response (simplified parsing)
            sentiment_score = self._extract_sentiment_score(content)
            direction = self._score_to_direction(sentiment_score)
            
            signal = Signal(
                signal_type=SignalType.SENTIMENT,
                source=SignalSource.GROK_X,
                asset_scope=AssetScope(tickers=[ticker]),
                summary=self._extract_summary(content),
                raw_value=RawValue(
                    type="index",
                    value=sentiment_score,
                    unit="sentiment_score",
                    prior_value=None,
                    change=None,
                    change_period=f"{lookback_hours}h"
                ),
                evidence=[
                    Evidence(
                        source="X/Twitter via Grok",
                        source_tier="social",
                        excerpt=content[:500],
                        timestamp_utc=datetime.now(timezone.utc)
                    )
                ],
                confidence=0.6,  # Social media is inherently uncertain
                directional_bias=direction,
                time_horizon=TimeHorizon.DAYS,
                novelty="new",
                timestamp_utc=datetime.now(timezone.utc),
                staleness_seconds=Config.STALENESS_SENTIMENT,
                uncertainties=["Social sentiment can shift rapidly", "Sample may not be representative"]
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Grok sentiment scan for {ticker}: {e}")
            return None
    
    async def scan_news(
        self,
        topics: List[str],
        lookback_hours: int = 24
    ) -> List[Signal]:
        """
        Scan web for news on given topics.
        
        Args:
            topics: List of topics/tickers to scan
            lookback_hours: How far back to look
            
        Returns:
            List of Signal objects
        """
        if not self.api_key:
            logger.warning("Grok scanner not configured - returning empty signals")
            return []
        
        signals = []
        
        for topic in topics:
            try:
                topic_signals = await self._scan_topic_news(topic, lookback_hours)
                signals.extend(topic_signals)
            except Exception as e:
                logger.error(f"Error scanning news for {topic}: {e}")
        
        return signals
    
    async def _scan_topic_news(
        self,
        topic: str,
        lookback_hours: int
    ) -> List[Signal]:
        """Scan news for a single topic."""
        
        system_prompt = """You are a financial news analyst. Search for and analyze recent news about the given topic.

You have access to web_search to find news articles. Focus on:
1. Breaking news that could move markets
2. Regulatory or policy changes
3. Corporate actions (earnings, M&A, management changes)
4. Sector trends
5. Macro events affecting the topic

For each significant news item, assess:
- Market impact (positive/negative/neutral)
- Time horizon (immediate to months)
- Confidence in the information
- Uncertainties or conflicting reports

Be factual. Cite sources. Distinguish between confirmed news and rumors."""

        user_prompt = f"""Find and analyze recent news about: {topic}

Focus on news from the last {lookback_hours} hours that could be market-moving.

For each significant item, provide:
1. What happened (factual summary)
2. Market implications
3. Affected assets
4. Source reliability
5. What's still uncertain"""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for news and information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        try:
            response = await self._call_grok(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=tools
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Create a signal from the news
            signal = Signal(
                signal_type=SignalType.EVENT,
                source=SignalSource.GROK_WEB,
                asset_scope=AssetScope(
                    tickers=[topic] if topic.isupper() and len(topic) <= 5 else [],
                    sectors=[topic] if not topic.isupper() else []
                ),
                summary=self._extract_summary(content),
                evidence=[
                    Evidence(
                        source="Web search via Grok",
                        source_tier="tier2",
                        excerpt=content[:500],
                        timestamp_utc=datetime.now(timezone.utc)
                    )
                ],
                confidence=0.7,
                directional_bias=self._extract_direction(content),
                time_horizon=TimeHorizon.DAYS,
                novelty="new",
                timestamp_utc=datetime.now(timezone.utc),
                staleness_seconds=Config.STALENESS_NEWS,
                uncertainties=["News interpretation may vary", "Full context may not be available"]
            )
            
            return [signal]
            
        except Exception as e:
            logger.error(f"Error in Grok news scan for {topic}: {e}")
            return []
    
    async def scan_market_overview(self) -> List[Signal]:
        """Get a general market sentiment overview."""
        
        if not self.api_key:
            return []
        
        system_prompt = """You are a market analyst. Provide a comprehensive overview of current market sentiment.

Search X/Twitter and the web to understand:
1. Overall market mood (risk-on vs risk-off)
2. Sector rotations happening
3. Major themes driving discussion
4. Geopolitical factors affecting markets
5. Fed/central bank sentiment
6. Key economic data releases

Be specific about what you find. Cite sources."""

        user_prompt = """Provide a current market sentiment overview:

1. What's the overall mood on financial Twitter?
2. Which sectors are being discussed positively/negatively?
3. What macro themes are dominating?
4. Any significant news moving markets today?
5. What are traders worried about or excited about?

Search both X and web news to get a complete picture."""

        try:
            response = await self._call_grok(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[
                    {"type": "function", "function": {"name": "x_search", "description": "Search X", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
                    {"type": "function", "function": {"name": "web_search", "description": "Search web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
                ]
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            signal = Signal(
                signal_type=SignalType.SENTIMENT,
                source=SignalSource.GROK_X,
                asset_scope=AssetScope(
                    macro_regions=["US"],
                    asset_classes=["EQUITY"]
                ),
                summary=f"Market Overview: {self._extract_summary(content)}",
                evidence=[
                    Evidence(
                        source="Grok market scan",
                        source_tier="social",
                        excerpt=content[:500],
                        timestamp_utc=datetime.now(timezone.utc)
                    )
                ],
                confidence=0.5,
                directional_bias=self._extract_direction(content),
                time_horizon=TimeHorizon.DAYS,
                novelty="new",
                timestamp_utc=datetime.now(timezone.utc),
                staleness_seconds=Config.STALENESS_SENTIMENT,
                uncertainties=["Market sentiment is inherently subjective"]
            )
            
            return [signal]
            
        except Exception as e:
            logger.error(f"Error in market overview scan: {e}")
            return []
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _extract_sentiment_score(self, content: str) -> float:
        """Extract sentiment score from Grok's response."""
        # Simple heuristic - look for explicit scores or keywords
        content_lower = content.lower()
        
        # Look for explicit score mentions
        import re
        score_match = re.search(r'sentiment[:\s]+(-?\d+\.?\d*)', content_lower)
        if score_match:
            try:
                return max(-1.0, min(1.0, float(score_match.group(1))))
            except:
                pass
        
        # Keyword-based scoring
        bullish_words = ['bullish', 'positive', 'optimistic', 'buy', 'long', 'moon', 'pump', 'breakout']
        bearish_words = ['bearish', 'negative', 'pessimistic', 'sell', 'short', 'dump', 'crash', 'breakdown']
        
        bullish_count = sum(1 for word in bullish_words if word in content_lower)
        bearish_count = sum(1 for word in bearish_words if word in content_lower)
        
        if bullish_count + bearish_count == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / (bullish_count + bearish_count)
    
    def _score_to_direction(self, score: float) -> DirectionalBias:
        """Convert sentiment score to directional bias."""
        if score > 0.3:
            return DirectionalBias.POSITIVE
        elif score < -0.3:
            return DirectionalBias.NEGATIVE
        elif abs(score) <= 0.1:
            return DirectionalBias.UNCLEAR
        else:
            return DirectionalBias.MIXED
    
    def _extract_summary(self, content: str) -> str:
        """Extract a concise summary from Grok's response."""
        # Take first paragraph or first 200 chars
        paragraphs = content.split('\n\n')
        if paragraphs:
            summary = paragraphs[0].strip()
            if len(summary) > 200:
                summary = summary[:197] + "..."
            return summary
        return content[:200] + "..." if len(content) > 200 else content
    
    def _extract_direction(self, content: str) -> DirectionalBias:
        """Extract directional bias from content."""
        content_lower = content.lower()
        
        positive_indicators = ['positive', 'bullish', 'optimistic', 'growth', 'expansion', 'beat', 'exceed']
        negative_indicators = ['negative', 'bearish', 'pessimistic', 'decline', 'contraction', 'miss', 'below']
        
        pos_count = sum(1 for word in positive_indicators if word in content_lower)
        neg_count = sum(1 for word in negative_indicators if word in content_lower)
        
        if pos_count > neg_count + 2:
            return DirectionalBias.POSITIVE
        elif neg_count > pos_count + 2:
            return DirectionalBias.NEGATIVE
        elif pos_count > 0 and neg_count > 0:
            return DirectionalBias.MIXED
        else:
            return DirectionalBias.UNCLEAR
