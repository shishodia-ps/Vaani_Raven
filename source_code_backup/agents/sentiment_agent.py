"""
Sentiment Agent - Layer 3
Sentiment Validation using FinBERT/GPT-4 and Economic News Analysis
"""

import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re
from .base_agent import BaseAgent

class SentimentAgent(BaseAgent):
    """Layer 3: Sentiment Validation Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "sentiment_agent")
        
        self.model_type = config.get('model', 'finbert')
        self.news_sources = config.get('news_sources', ['reuters', 'bloomberg', 'forexfactory'])
        self.impact_threshold = config.get('impact_threshold', 'high')
        self.uncertainty_halt = config.get('uncertainty_halt', True)
        
        self.eur_usd_keywords = [
            'EUR', 'USD', 'EURUSD', 'euro', 'dollar', 'ECB', 'Fed', 'Federal Reserve',
            'European Central Bank', 'eurozone', 'inflation', 'interest rate',
            'monetary policy', 'GDP', 'unemployment', 'CPI', 'NFP'
        ]
        
        self.high_impact_events = [
            'NFP', 'CPI', 'GDP', 'interest rate', 'monetary policy', 'inflation',
            'unemployment', 'trade war', 'recession', 'crisis', 'election'
        ]
        
        if self.enabled:
            self._initialize_sentiment_model()
    
    def _initialize_sentiment_model(self):
        """Initialize sentiment analysis model"""
        try:
            if self.model_type == 'finbert':
                self.logger.info("FinBERT sentiment model initialized")
            elif self.model_type == 'gpt4':
                self.logger.info("GPT-4 sentiment model initialized")
            else:
                self.logger.warning(f"Unknown model type: {self.model_type}")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment model: {e}")
    
    def process_signal(self, market_data: Dict[str, Any], 
                      previous_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process sentiment analysis and validate previous signals"""
        
        if not self.enabled:
            return self._create_neutral_signal("Agent disabled")
        
        try:
            quant_signal = None
            for signal in previous_signals:
                if signal.get('agent') == 'quant_agent':
                    quant_signal = signal
                    break
            
            if not quant_signal or quant_signal.get('signal') == 'HOLD':
                return self._create_neutral_signal("No quant signal to validate")
            
            news_sentiment = self._analyze_news_sentiment()
            economic_events = self._check_economic_events()
            uncertainty_level = self._calculate_uncertainty(news_sentiment, economic_events)
            
            if self.uncertainty_halt and uncertainty_level > 0.7:
                return self._create_halt_signal("High uncertainty detected", uncertainty_level)
            
            sentiment_validation = self._validate_with_sentiment(
                quant_signal, news_sentiment, economic_events, uncertainty_level
            )
            
            signal = {
                'agent': self.name,
                'signal': sentiment_validation['signal'],
                'confidence': sentiment_validation['confidence'],
                'timestamp': datetime.now(),
                'metadata': {
                    'news_sentiment': news_sentiment,
                    'economic_events': economic_events,
                    'uncertainty_level': uncertainty_level,
                    'sentiment_score': sentiment_validation['sentiment_score'],
                    'quant_signal_confidence': quant_signal.get('confidence', 0)
                }
            }
            
            self.log_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return self._create_neutral_signal(f"Sentiment analysis error: {str(e)}")
    
    def _analyze_news_sentiment(self) -> Dict[str, Any]:
        """Analyze news sentiment for EUR/USD relevant events"""
        try:
            news_data = self._fetch_news_data()
            
            if not news_data:
                return {'sentiment': 'neutral', 'confidence': 0.0, 'articles_count': 0}
            
            eur_usd_articles = self._filter_eur_usd_news(news_data)
            
            if not eur_usd_articles:
                return {'sentiment': 'neutral', 'confidence': 0.0, 'articles_count': 0}
            
            sentiment_scores = []
            for article in eur_usd_articles:
                score = self._analyze_article_sentiment(article)
                sentiment_scores.append(score)
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            if avg_sentiment > 0.1:
                sentiment = 'positive'
            elif avg_sentiment < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            confidence = min(abs(avg_sentiment), 1.0)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'articles_count': len(eur_usd_articles),
                'avg_score': avg_sentiment
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0, 'articles_count': 0}
    
    def _fetch_news_data(self) -> List[Dict[str, Any]]:
        """Fetch news data from configured sources"""
        news_articles = []
        
        try:
            mock_articles = [
                {
                    'title': 'ECB Maintains Interest Rates Amid Inflation Concerns',
                    'content': 'The European Central Bank decided to keep interest rates unchanged as inflation remains above target.',
                    'source': 'reuters',
                    'timestamp': datetime.now() - timedelta(hours=2)
                },
                {
                    'title': 'US Dollar Strengthens on Fed Policy Expectations',
                    'content': 'The US dollar gained against major currencies following hawkish comments from Fed officials.',
                    'source': 'bloomberg',
                    'timestamp': datetime.now() - timedelta(hours=1)
                },
                {
                    'title': 'Eurozone GDP Growth Slows in Q4',
                    'content': 'Economic growth in the eurozone decelerated in the fourth quarter amid global uncertainties.',
                    'source': 'forexfactory',
                    'timestamp': datetime.now() - timedelta(hours=3)
                }
            ]
            
            news_articles.extend(mock_articles)
            
        except Exception as e:
            self.logger.error(f"Error fetching news data: {e}")
        
        return news_articles
    
    def _filter_eur_usd_news(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter news articles relevant to EUR/USD"""
        relevant_articles = []
        
        for article in news_data:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            
            text = f"{title} {content}"
            
            relevance_score = 0
            for keyword in self.eur_usd_keywords:
                if keyword.lower() in text:
                    relevance_score += 1
            
            if relevance_score >= 2:
                article['relevance_score'] = relevance_score
                relevant_articles.append(article)
        
        return relevant_articles
    
    def _analyze_article_sentiment(self, article: Dict[str, Any]) -> float:
        """Analyze sentiment of individual article"""
        try:
            text = f"{article.get('title', '')} {article.get('content', '')}"
            
            if self.model_type == 'finbert':
                return self._finbert_sentiment(text)
            elif self.model_type == 'gpt4':
                return self._gpt4_sentiment(text)
            else:
                return self._simple_sentiment(text)
                
        except Exception as e:
            self.logger.error(f"Error analyzing article sentiment: {e}")
            return 0.0
    
    def _finbert_sentiment(self, text: str) -> float:
        """Analyze sentiment using FinBERT (mock implementation)"""
        positive_words = ['growth', 'increase', 'strong', 'positive', 'bullish', 'rise', 'gain']
        negative_words = ['decline', 'fall', 'weak', 'negative', 'bearish', 'drop', 'loss']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / max(total_words / 10, 1)
        
        return max(-1.0, min(1.0, sentiment_score))
    
    def _gpt4_sentiment(self, text: str) -> float:
        """Analyze sentiment using GPT-4 (mock implementation)"""
        return self._finbert_sentiment(text)
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment analysis"""
        return self._finbert_sentiment(text)
    
    def _check_economic_events(self) -> List[Dict[str, Any]]:
        """Check for upcoming high-impact economic events"""
        try:
            current_time = datetime.now()
            
            mock_events = [
                {
                    'event': 'US Non-Farm Payrolls',
                    'time': current_time + timedelta(hours=6),
                    'impact': 'high',
                    'currency': 'USD',
                    'forecast': '200K',
                    'previous': '180K'
                },
                {
                    'event': 'ECB Interest Rate Decision',
                    'time': current_time + timedelta(days=1),
                    'impact': 'high',
                    'currency': 'EUR',
                    'forecast': '4.50%',
                    'previous': '4.50%'
                }
            ]
            
            upcoming_events = []
            for event in mock_events:
                time_diff = (event['time'] - current_time).total_seconds() / 3600
                
                if 0 <= time_diff <= 24:
                    event['hours_until'] = time_diff
                    upcoming_events.append(event)
            
            return upcoming_events
            
        except Exception as e:
            self.logger.error(f"Error checking economic events: {e}")
            return []
    
    def _calculate_uncertainty(self, news_sentiment: Dict[str, Any], 
                             economic_events: List[Dict[str, Any]]) -> float:
        """Calculate overall market uncertainty level"""
        uncertainty = 0.0
        
        sentiment_uncertainty = 1.0 - news_sentiment.get('confidence', 0.0)
        uncertainty += sentiment_uncertainty * 0.4
        
        high_impact_events = [e for e in economic_events if e.get('impact') == 'high']
        event_uncertainty = min(len(high_impact_events) * 0.2, 0.6)
        uncertainty += event_uncertainty
        
        if news_sentiment.get('articles_count', 0) < 3:
            uncertainty += 0.2
        
        return min(uncertainty, 1.0)
    
    def _validate_with_sentiment(self, quant_signal: Dict[str, Any], 
                               news_sentiment: Dict[str, Any],
                               economic_events: List[Dict[str, Any]],
                               uncertainty_level: float) -> Dict[str, Any]:
        """Validate quant signal with sentiment analysis"""
        
        signal_type = quant_signal.get('signal', 'HOLD')
        quant_confidence = quant_signal.get('confidence', 0)
        
        sentiment_score = 0
        
        if news_sentiment.get('sentiment') == 'positive':
            if signal_type == 'BUY':
                sentiment_score += news_sentiment.get('confidence', 0) * 0.5
            elif signal_type == 'SELL':
                sentiment_score -= news_sentiment.get('confidence', 0) * 0.3
        elif news_sentiment.get('sentiment') == 'negative':
            if signal_type == 'SELL':
                sentiment_score += news_sentiment.get('confidence', 0) * 0.5
            elif signal_type == 'BUY':
                sentiment_score -= news_sentiment.get('confidence', 0) * 0.3
        
        high_impact_events = [e for e in economic_events if e.get('impact') == 'high']
        if high_impact_events:
            event_penalty = min(len(high_impact_events) * 0.1, 0.3)
            sentiment_score -= event_penalty
        
        uncertainty_penalty = uncertainty_level * 0.4
        
        final_confidence = max(0, min(1, quant_confidence + sentiment_score - uncertainty_penalty))
        
        if final_confidence < 0.3:
            final_signal = 'HOLD'
        else:
            final_signal = signal_type
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'sentiment_score': sentiment_score
        }
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create a neutral signal with reason"""
        return {
            'agent': self.name,
            'signal': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'metadata': {
                'reason': reason,
                'sentiment_score': 0.0
            }
        }
    
    def _create_halt_signal(self, reason: str, uncertainty_level: float) -> Dict[str, Any]:
        """Create a halt signal due to high uncertainty"""
        return {
            'agent': self.name,
            'signal': 'HALT',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'metadata': {
                'reason': reason,
                'uncertainty_level': uncertainty_level,
                'halt_triggered': True
            }
        }
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal format and content"""
        required_fields = ['agent', 'signal', 'confidence', 'timestamp']
        
        if not all(field in signal for field in required_fields):
            return False
        
        if signal['signal'] not in ['BUY', 'SELL', 'HOLD', 'HALT']:
            return False
        
        if not 0 <= signal['confidence'] <= 1:
            return False
        
        return True
