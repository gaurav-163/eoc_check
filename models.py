"""
Pydantic models for VAPI EOC Fetcher
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============== Enums ==============
class CallStatus(str, Enum):
    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    FORWARDING = "forwarding"
    ENDED = "ended"


class EndedReason(str, Enum):
    ASSISTANT_ENDED = "assistant-ended-call"
    CUSTOMER_ENDED = "customer-ended-call"
    SILENCE_TIMEOUT = "silence-timed-out"
    MAX_DURATION = "max-duration-reached"
    ERROR = "error"
    VOICEMAIL = "voicemail-detected"
    PIPELINE_ERROR = "pipeline-error"
    CUSTOMER_BUSY = "customer-busy"
    CUSTOMER_NO_ANSWER = "customer-no-answer"


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


# ============== Message Models ==============
class TranscriptMessage(BaseModel):
    role: str = Field(..., description="Speaker role: user/assistant/system")
    message: str = Field(..., description="Message content")
    timestamp: Optional[float] = Field(None, description="Time in seconds")
    duration: Optional[float] = Field(None, description="Speech duration")
    
    class Config:
        extra = "allow"


class Transcript(BaseModel):
    messages: List[TranscriptMessage] = Field(default_factory=list)
    full_text: Optional[str] = None
    
    def get_full_text(self) -> str:
        """Combine all messages into full text"""
        if self.full_text:
            return self.full_text
        return "\n".join([
            f"{msg.role}: {msg.message}" 
            for msg in self.messages
        ])


# ============== Call Cost Models ==============
class CostBreakdown(BaseModel):
    stt: Optional[float] = Field(0.0, description="Speech-to-text cost")
    llm: Optional[float] = Field(0.0, description="LLM cost")
    tts: Optional[float] = Field(0.0, description="Text-to-speech cost")
    vapi: Optional[float] = Field(0.0, description="VAPI platform cost")
    transport: Optional[float] = Field(0.0, description="Transport cost")
    total: Optional[float] = Field(0.0, description="Total cost")
    
    class Config:
        extra = "allow"


# ============== Analysis Models ==============
class SentimentAnalysis(BaseModel):
    overall: SentimentType = SentimentType.NEUTRAL
    score: float = Field(0.0, ge=-1.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    segments: List[Dict[str, Any]] = Field(default_factory=list)


class KeyTopics(BaseModel):
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: List[Dict[str, str]] = Field(default_factory=list)


class CallAnalysis(BaseModel):
    sentiment: SentimentAnalysis
    topics: KeyTopics
    summary: str = ""
    action_items: List[str] = Field(default_factory=list)
    customer_intent: Optional[str] = None
    resolution_status: Optional[str] = None
    talk_ratio: Dict[str, float] = Field(default_factory=dict)
    word_count: int = 0
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


# ============== VAPI Call Models ==============
class VAPICallMetadata(BaseModel):
    assistant_id: Optional[str] = None
    assistant_name: Optional[str] = None
    phone_number: Optional[str] = None
    customer_phone: Optional[str] = None
    squad_id: Optional[str] = None
    
    class Config:
        extra = "allow"


class VAPICall(BaseModel):
    """Main VAPI Call model"""
    id: str = Field(..., description="Unique call ID")
    org_id: Optional[str] = None
    type: str = Field("webCall", description="Call type")
    status: CallStatus = CallStatus.ENDED
    ended_reason: Optional[EndedReason] = None
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Duration
    duration_seconds: Optional[float] = None
    
    # Content
    transcript: Optional[str] = None
    messages: List[TranscriptMessage] = Field(default_factory=list)
    recording_url: Optional[str] = None
    stereo_recording_url: Optional[str] = None
    
    # Cost
    cost: Optional[float] = None
    cost_breakdown: Optional[CostBreakdown] = None
    
    # Metadata
    metadata: Optional[VAPICallMetadata] = None
    analysis: Optional[CallAnalysis] = None
    
    # Raw data
    raw_data: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"
        
    @property
    def duration_minutes(self) -> float:
        """Get duration in minutes"""
        if self.duration_seconds:
            return round(self.duration_seconds / 60, 2)
        return 0.0
    
    def get_transcript_object(self) -> Transcript:
        """Get structured transcript"""
        return Transcript(
            messages=self.messages,
            full_text=self.transcript
        )


# ============== API Request/Response Models ==============
class FetchCallsRequest(BaseModel):
    limit: int = Field(100, ge=1, le=1000)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    assistant_id: Optional[str] = None
    status: Optional[CallStatus] = None


class FetchCallsResponse(BaseModel):
    success: bool
    total_calls: int
    calls: List[VAPICall]
    errors: List[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class AnalyzeRequest(BaseModel):
    call_id: str
    include_sentiment: bool = True
    include_topics: bool = True
    include_summary: bool = True
    include_action_items: bool = True


class AnalyzeResponse(BaseModel):
    success: bool
    call_id: str
    analysis: Optional[CallAnalysis] = None
    error: Optional[str] = None


class WebhookPayload(BaseModel):
    """VAPI Webhook end-of-call payload"""
    message: Dict[str, Any]
    
    class Config:
        extra = "allow"


class EOCReportMessage(BaseModel):
    """End of Call Report structure from VAPI webhook"""
    type: str = "end-of-call-report"
    call: Dict[str, Any]
    ended_reason: Optional[str] = None
    transcript: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Optional[str] = None
    recording_url: Optional[str] = None
    
    class Config:
        extra = "allow"


# ============== Dashboard/Stats Models ==============
class CallStats(BaseModel):
    total_calls: int = 0
    total_duration_minutes: float = 0.0
    average_duration_minutes: float = 0.0
    total_cost: float = 0.0
    average_cost: float = 0.0
    
    calls_by_status: Dict[str, int] = Field(default_factory=dict)
    calls_by_ended_reason: Dict[str, int] = Field(default_factory=dict)
    calls_by_sentiment: Dict[str, int] = Field(default_factory=dict)
    
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class DailyStats(BaseModel):
    date: str
    call_count: int
    total_duration: float
    total_cost: float
    avg_sentiment_score: float


class AnalyticsDashboard(BaseModel):
    stats: CallStats
    daily_breakdown: List[DailyStats] = Field(default_factory=list)
    top_topics: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_trend: List[Dict[str, Any]] = Field(default_factory=list)