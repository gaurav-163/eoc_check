"""
Database operations for storing and retrieving calls
"""
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from config import settings
from models import VAPICall, CallAnalysis, CallStats

# Database setup
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============== Database Models ==============
class CallRecord(Base):
    """SQLAlchemy model for call storage"""
    __tablename__ = "calls"
    
    id = Column(String, primary_key=True, index=True)
    org_id = Column(String, nullable=True, index=True)
    assistant_id = Column(String, nullable=True, index=True)
    call_type = Column(String, default="webCall")
    status = Column(String, index=True)
    ended_reason = Column(String, nullable=True)
    
    created_at = Column(DateTime, index=True)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    
    duration_seconds = Column(Float, default=0.0)
    cost = Column(Float, default=0.0)
    
    transcript = Column(Text, nullable=True)
    messages_json = Column(Text, nullable=True)
    recording_url = Column(String, nullable=True)
    
    analysis_json = Column(Text, nullable=True)
    raw_data_json = Column(Text, nullable=True)
    
    # Metadata
    customer_phone = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    
    inserted_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WebhookLog(Base):
    """Log of received webhooks"""
    __tablename__ = "webhook_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String, nullable=True, index=True)
    event_type = Column(String, index=True)
    payload_json = Column(Text)
    received_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Integer, default=0)


# Create tables
Base.metadata.create_all(bind=engine)


# ============== Database Session ==============
@contextmanager
def get_db_session() -> Session:
    """Get database session context manager"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============== CRUD Operations ==============
class CallRepository:
    """Repository for call operations"""
    
    @staticmethod
    def save_call(db: Session, call: VAPICall) -> CallRecord:
        """Save or update a call record"""
        existing = db.query(CallRecord).filter(CallRecord.id == call.id).first()
        
        if existing:
            # Update existing
            existing.status = call.status.value if call.status else None
            existing.ended_reason = call.ended_reason.value if call.ended_reason else None
            existing.ended_at = call.ended_at
            existing.duration_seconds = call.duration_seconds
            existing.cost = call.cost
            existing.transcript = call.transcript
            existing.messages_json = json.dumps([m.dict() for m in call.messages]) if call.messages else None
            existing.recording_url = call.recording_url
            existing.analysis_json = json.dumps(call.analysis.dict()) if call.analysis else None
            existing.raw_data_json = json.dumps(call.raw_data) if call.raw_data else None
            existing.updated_at = datetime.utcnow()
            return existing
        
        # Create new
        record = CallRecord(
            id=call.id,
            org_id=call.org_id,
            assistant_id=call.metadata.assistant_id if call.metadata else None,
            call_type=call.type,
            status=call.status.value if call.status else None,
            ended_reason=call.ended_reason.value if call.ended_reason else None,
            created_at=call.created_at,
            started_at=call.started_at,
            ended_at=call.ended_at,
            duration_seconds=call.duration_seconds,
            cost=call.cost,
            transcript=call.transcript,
            messages_json=json.dumps([m.dict() for m in call.messages]) if call.messages else None,
            recording_url=call.recording_url,
            customer_phone=call.metadata.customer_phone if call.metadata else None,
            raw_data_json=json.dumps(call.raw_data) if call.raw_data else None,
        )
        db.add(record)
        return record
    
    @staticmethod
    def get_call(db: Session, call_id: str) -> Optional[VAPICall]:
        """Get a call by ID"""
        record = db.query(CallRecord).filter(CallRecord.id == call_id).first()
        if not record:
            return None
        return CallRepository._record_to_model(record)
    
    @staticmethod
    def get_calls(
        db: Session,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        assistant_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[VAPICall]:
        """Get calls with filters"""
        query = db.query(CallRecord)
        
        if start_date:
            query = query.filter(CallRecord.created_at >= start_date)
        if end_date:
            query = query.filter(CallRecord.created_at <= end_date)
        if assistant_id:
            query = query.filter(CallRecord.assistant_id == assistant_id)
        if status:
            query = query.filter(CallRecord.status == status)
        
        records = query.order_by(CallRecord.created_at.desc()).offset(offset).limit(limit).all()
        return [CallRepository._record_to_model(r) for r in records]
    
    @staticmethod
    def get_stats(
        db: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> CallStats:
        """Get call statistics"""
        from sqlalchemy import func
        
        query = db.query(CallRecord)
        if start_date:
            query = query.filter(CallRecord.created_at >= start_date)
        if end_date:
            query = query.filter(CallRecord.created_at <= end_date)
        
        records = query.all()
        
        if not records:
            return CallStats()
        
        total_duration = sum(r.duration_seconds or 0 for r in records)
        total_cost = sum(r.cost or 0 for r in records)
        
        calls_by_status = {}
        calls_by_reason = {}
        
        for r in records:
            if r.status:
                calls_by_status[r.status] = calls_by_status.get(r.status, 0) + 1
            if r.ended_reason:
                calls_by_reason[r.ended_reason] = calls_by_reason.get(r.ended_reason, 0) + 1
        
        return CallStats(
            total_calls=len(records),
            total_duration_minutes=round(total_duration / 60, 2),
            average_duration_minutes=round((total_duration / len(records)) / 60, 2) if records else 0,
            total_cost=round(total_cost, 4),
            average_cost=round(total_cost / len(records), 4) if records else 0,
            calls_by_status=calls_by_status,
            calls_by_ended_reason=calls_by_reason,
            date_range_start=start_date,
            date_range_end=end_date
        )
    
    @staticmethod
    def save_analysis(db: Session, call_id: str, analysis: CallAnalysis) -> bool:
        """Save analysis for a call"""
        record = db.query(CallRecord).filter(CallRecord.id == call_id).first()
        if not record:
            return False
        
        record.analysis_json = json.dumps(analysis.dict(), default=str)
        record.sentiment_score = analysis.sentiment.score
        record.updated_at = datetime.utcnow()
        return True
    
    @staticmethod
    def _record_to_model(record: CallRecord) -> VAPICall:
        """Convert database record to Pydantic model"""
        from models import TranscriptMessage, VAPICallMetadata, CallStatus, EndedReason
        
        messages = []
        if record.messages_json:
            try:
                messages = [TranscriptMessage(**m) for m in json.loads(record.messages_json)]
            except:
                pass
        
        analysis = None
        if record.analysis_json:
            try:
                analysis = CallAnalysis(**json.loads(record.analysis_json))
            except:
                pass
        
        raw_data = None
        if record.raw_data_json:
            try:
                raw_data = json.loads(record.raw_data_json)
            except:
                pass
        
        return VAPICall(
            id=record.id,
            org_id=record.org_id,
            type=record.call_type,
            status=CallStatus(record.status) if record.status else CallStatus.ENDED,
            ended_reason=EndedReason(record.ended_reason) if record.ended_reason else None,
            created_at=record.created_at,
            started_at=record.started_at,
            ended_at=record.ended_at,
            duration_seconds=record.duration_seconds,
            transcript=record.transcript,
            messages=messages,
            recording_url=record.recording_url,
            cost=record.cost,
            metadata=VAPICallMetadata(
                assistant_id=record.assistant_id,
                customer_phone=record.customer_phone
            ),
            analysis=analysis,
            raw_data=raw_data
        )


# ============== Webhook Logging ==============
class WebhookRepository:
    """Repository for webhook operations"""
    
    @staticmethod
    def log_webhook(db: Session, event_type: str, payload: Dict[str, Any], call_id: str = None):
        """Log a webhook event"""
        log = WebhookLog(
            call_id=call_id,
            event_type=event_type,
            payload_json=json.dumps(payload, default=str)
        )
        db.add(log)
        return log
    
    @staticmethod
    def mark_processed(db: Session, log_id: int):
        """Mark webhook as processed"""
        log = db.query(WebhookLog).filter(WebhookLog.id == log_id).first()
        if log:
            log.processed = 1