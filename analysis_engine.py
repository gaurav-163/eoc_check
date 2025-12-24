"""
Transcript analysis engine using NLP and optional LLM
"""
import re
import logging
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime

from models import (
    VAPICall, CallAnalysis, SentimentAnalysis, KeyTopics,
    TranscriptMessage, SentimentType, EndedReason
)
from config import settings

logger = logging.getLogger(__name__)


# ============== Simple Sentiment Analyzer ==============
class SimpleSentimentAnalyzer:
    """Rule-based sentiment analysis (no dependencies)"""
    
    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'happy', 'thank', 'thanks', 'appreciate', 'helpful',
        'perfect', 'awesome', 'best', 'pleased', 'satisfied', 'yes',
        'absolutely', 'definitely', 'sure', 'right', 'agree', 'nice',
        'brilliant', 'delightful', 'fantastic', 'outstanding', 'superb',
        'terrific', 'wonderful', 'exceptional', 'fabulous', 'marvelous'
    }
    
    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'upset',
        'disappointed', 'frustrated', 'annoying', 'worst', 'never', 'no',
        'wrong', 'problem', 'issue', 'error', 'fail', 'failed', 'broken',
        'useless', 'waste', 'stupid', 'ridiculous', 'unacceptable',
        'dreadful', 'lousy', 'poor', 'inferior', 'substandard',
        'unsatisfactory', 'mediocre', 'awful', 'atrocious', 'appalling'
    }
    
    INTENSIFIERS = {'very', 'really', 'extremely', 'absolutely', 'totally'}
    NEGATORS = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing'}
    
    def analyze(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment of text"""
        if not text:
            return SentimentAnalysis(
                overall=SentimentType.NEUTRAL,
                score=0.0,
                confidence=0.0
            )
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = 0
        negative_count = 0
        total_words = len(words)
        
        for i, word in enumerate(words):
            # Check for negation
            negated = False
            if i > 0 and words[i-1] in self.NEGATORS:
                negated = True
            
            # Check intensifier
            intensified = 1.0
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                intensified = 1.5
            
            if word in self.POSITIVE_WORDS:
                if negated:
                    negative_count += intensified
                else:
                    positive_count += intensified
            elif word in self.NEGATIVE_WORDS:
                if negated:
                    positive_count += intensified
                else:
                    negative_count += intensified
        
        # Calculate score (-1 to 1)
        if positive_count + negative_count == 0:
            score = 0.0
            confidence = 0.3
        else:
            score = (positive_count - negative_count) / (positive_count + negative_count)
            confidence = min(1.0, (positive_count + negative_count) / (total_words * 0.1))
        
        # Determine overall sentiment
        if score > 0.2:
            overall = SentimentType.POSITIVE
        elif score < -0.2:
            overall = SentimentType.NEGATIVE
        else:
            overall = SentimentType.NEUTRAL
        
        return SentimentAnalysis(
            overall=overall,
            score=round(score, 3),
            confidence=round(confidence, 3)
        )


# ============== Topic Extractor ==============
class TopicExtractor:
    """Extract key topics and keywords from text"""
    
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
        'um', 'uh', 'like', 'okay', 'ok', 'yeah', 'yes', 'no', 'right',
        'well', 'so', 'just', 'really', 'actually', 'basically', 'literally',
        'also', 'always', 'never', 'ever', 'still', 'yet', 'already',
        'almost', 'quite', 'rather', 'somewhat', 'therefore', 'hence',
        'thus', 'however', 'although', 'though', 'even', 'only', 'just'
    }
    
    # Common business/support topics
    TOPIC_PATTERNS = {
        'billing': r'\b(bill|billing|payment|charge|invoice|refund|price|cost|fee|rate|subscription)\b',
        'account': r'\b(account|login|password|username|profile|settings|dashboard|preferences)\b',
        'technical': r'\b(error|bug|crash|issue|problem|broken|not working|fix|glitch|malfunction)\b',
        'shipping': r'\b(shipping|delivery|order|package|track|arrive|ship|deliver|logistics)\b',
        'product': r'\b(product|item|feature|quality|size|color|specification|attribute|characteristic)\b',
        'cancellation': r'\b(cancel|cancellation|stop|end|terminate|abort|revoke|withdraw)\b',
        'appointment': r'\b(appointment|schedule|booking|meeting|call back|reservation|slot|time)\b',
        'information': r'\b(information|info|details|question|help|how to|guide|manual|documentation)\b',
        'support': r'\b(support|assistance|helpdesk|customer service|troubleshoot|resolve)\b',
        'feedback': r'\b(feedback|review|comment|opinion|suggestion|complaint|praise)\b',
    }
    
    def extract(self, text: str) -> KeyTopics:
        """Extract topics and keywords from text"""
        if not text:
            return KeyTopics()
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Filter stop words and short words
        filtered_words = [
            w for w in words 
            if w not in self.STOP_WORDS and len(w) > 2
        ]
        
        # Get word frequencies
        word_freq = Counter(filtered_words)
        keywords = [word for word, count in word_freq.most_common(10)]
        
        # Match topic patterns
        topics = []
        for topic, pattern in self.TOPIC_PATTERNS.items():
            if re.search(pattern, text_lower):
                topics.append(topic)
        
        # Extract potential entities (capitalized words, excluding sentence starts)
        sentences = text.split('.')
        entities = []
        for sentence in sentences:
            words_in_sentence = sentence.strip().split()
            for i, word in enumerate(words_in_sentence[1:], 1):  # Skip first word
                if word and word[0].isupper() and word.lower() not in self.STOP_WORDS:
                    entities.append({
                        "text": word,
                        "type": "UNKNOWN"
                    })
        
        # Deduplicate entities
        seen = set()
        unique_entities = []
        for e in entities:
            if e["text"] not in seen:
                seen.add(e["text"])
                unique_entities.append(e)
        
        return KeyTopics(
            topics=topics[:5],
            keywords=keywords,
            entities=unique_entities[:10]
        )


# ============== LLM Analyzer (Optional) ==============
class LLMAnalyzer:
    """LLM-based analysis using OpenAI"""
    
    def __init__(self):
        self.enabled = bool(settings.OPENAI_API_KEY)
        self._client = None
    
    @property
    def client(self):
        if not self._client and self.enabled:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            except ImportError:
                logger.warning("OpenAI package not installed")
                self.enabled = False
        return self._client
    
    async def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """Analyze transcript using LLM"""
        if not self.enabled or not self.client:
            return {}
        
        try:
            prompt = f"""Analyze this call transcript and provide:
1. A brief summary (2-3 sentences)
2. The main customer intent
3. Resolution status (resolved/unresolved/partial/escalated)
4. Any action items mentioned
5. Overall sentiment (positive/negative/neutral)

Transcript:
{transcript[:4000]}  # Limit transcript length

Respond in JSON format:
{{
    "summary": "...",
    "customer_intent": "...",
    "resolution_status": "...",
    "action_items": ["..."],
    "sentiment": "..."
}}"""

            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a call analysis assistant. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {}


# ============== Main Analysis Engine ==============
class AnalysisEngine:
    """Main analysis engine combining all analyzers"""
    
    def __init__(self, use_llm: bool = True):
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.topic_extractor = TopicExtractor()
        self.llm_analyzer = LLMAnalyzer() if use_llm else None
    
    def analyze_call(self, call: VAPICall, use_llm: bool = False) -> CallAnalysis:
        """
        Analyze a call synchronously (rule-based only)
        """
        transcript = call.get_transcript_object()
        full_text = transcript.get_full_text()
        
        # Basic analysis
        sentiment = self.sentiment_analyzer.analyze(full_text)
        topics = self.topic_extractor.extract(full_text)
        
        # Calculate talk ratio
        talk_ratio = self._calculate_talk_ratio(call.messages)
        
        # Word count
        word_count = len(full_text.split()) if full_text else 0
        
        # Generate basic summary
        summary = self._generate_basic_summary(call, sentiment, topics)
        
        return CallAnalysis(
            sentiment=sentiment,
            topics=topics,
            summary=summary,
            action_items=[],
            customer_intent=topics.topics[0] if topics.topics else None,
            resolution_status=self._infer_resolution(call),
            talk_ratio=talk_ratio,
            word_count=word_count
        )

    async def analyze_calls_batch_async(
        self,
        calls: List[VAPICall],
        use_llm: bool = False,
        batch_size: int = 1000,
        max_concurrency: int = 10
    ) -> List[CallAnalysis]:
        """
        Analyze multiple calls asynchronously with controlled concurrency for large datasets
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_call(call: VAPICall) -> CallAnalysis:
            async with semaphore:
                if use_llm:
                    return await self.analyze_call_async(call)
                else:
                    return self.analyze_call(call)
        
        for i in range(0, len(calls), batch_size):
            batch = calls[i:i + batch_size]
            batch_results = await asyncio.gather(*[process_call(call) for call in batch])
            results.extend(batch_results)
        
        return results
    
    async def analyze_call_async(self, call: VAPICall) -> CallAnalysis:
        """
        Analyze a call with optional LLM enhancement
        """
        # Start with basic analysis
        analysis = self.analyze_call(call, use_llm=False)
        
        # Enhance with LLM if available
        if self.llm_analyzer and self.llm_analyzer.enabled:
            transcript = call.get_transcript_object()
            full_text = transcript.get_full_text()
            
            if full_text and len(full_text) > 50:  # Minimum text length
                llm_result = await self.llm_analyzer.analyze_transcript(full_text)
                
                if llm_result:
                    # Update analysis with LLM results
                    if llm_result.get("summary"):
                        analysis.summary = llm_result["summary"]
                    if llm_result.get("customer_intent"):
                        analysis.customer_intent = llm_result["customer_intent"]
                    if llm_result.get("resolution_status"):
                        analysis.resolution_status = llm_result["resolution_status"]
                    if llm_result.get("action_items"):
                        analysis.action_items = llm_result["action_items"]
                    if llm_result.get("sentiment"):
                        sentiment_map = {
                            "positive": SentimentType.POSITIVE,
                            "negative": SentimentType.NEGATIVE,
                            "neutral": SentimentType.NEUTRAL
                        }
                        if llm_result["sentiment"].lower() in sentiment_map:
                            analysis.sentiment.overall = sentiment_map[llm_result["sentiment"].lower()]
        
        return analysis
    
    def _calculate_talk_ratio(self, messages: List[TranscriptMessage]) -> Dict[str, float]:
        """Calculate speaking time ratio between participants"""
        if not messages:
            return {}
        
        word_counts = {}
        for msg in messages:
            role = msg.role
            words = len(msg.message.split()) if msg.message else 0
            word_counts[role] = word_counts.get(role, 0) + words
        
        total = sum(word_counts.values())
        if total == 0:
            return {}
        
        return {
            role: round(count / total * 100, 1)
            for role, count in word_counts.items()
        }
    
    def _generate_basic_summary(
        self, 
        call: VAPICall, 
        sentiment: SentimentAnalysis, 
        topics: KeyTopics
    ) -> str:
        """Generate a basic summary without LLM"""
        parts = []
        
        # Duration info
        if call.duration_seconds:
            minutes = int(call.duration_seconds // 60)
            seconds = int(call.duration_seconds % 60)
            parts.append(f"Call lasted {minutes}m {seconds}s.")
        
        # Topics
        if topics.topics:
            parts.append(f"Main topics: {', '.join(topics.topics)}.")
        
        # Sentiment
        parts.append(f"Overall sentiment: {sentiment.overall.value}.")
        
        # Ended reason
        if call.ended_reason:
            reason = call.ended_reason.value.replace("-", " ")
            parts.append(f"Call ended: {reason}.")
        
        return " ".join(parts)
    
    def _infer_resolution(self, call: VAPICall) -> str:
        """Infer resolution status from call data"""
        if call.ended_reason == EndedReason.ASSISTANT_ENDED:
            return "likely_resolved"
        elif call.ended_reason == EndedReason.CUSTOMER_ENDED:
            if call.duration_seconds and call.duration_seconds > 60:
                return "likely_resolved"
            return "unknown"
        elif call.ended_reason in [EndedReason.ERROR, EndedReason.PIPELINE_ERROR]:
            return "error"
        elif call.ended_reason == EndedReason.SILENCE_TIMEOUT:
            return "abandoned"
        return "unknown"


# ============== Batch Analysis ==============
async def analyze_calls_batch(
    calls: List[VAPICall],
    use_llm: bool = False,
    batch_size: int = 1000
) -> List[CallAnalysis]:
    """Analyze multiple calls in batches for better performance"""
    engine = AnalysisEngine(use_llm=use_llm)
    results = []
    
    for i in range(0, len(calls), batch_size):
        batch = calls[i:i + batch_size]
        batch_results = []
        
        if use_llm:
            batch_results = await asyncio.gather(*[engine.analyze_call_async(call) for call in batch])
        else:
            for call in batch:
                analysis = engine.analyze_call(call)
                batch_results.append(analysis)
        
        results.extend(batch_results)
    
    return results


# ============== Convenience Functions ==============
def quick_analyze(call: VAPICall) -> CallAnalysis:
    """Quick synchronous analysis"""
    engine = AnalysisEngine(use_llm=False)
    return engine.analyze_call(call)