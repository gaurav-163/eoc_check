"""
VAPI EOC Fetcher - Main FastAPI Application
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from config import settings
from models import (
    VAPICall, CallAnalysis, CallStats, CallStatus,
    FetchCallsRequest, FetchCallsResponse,
    AnalyzeRequest, AnalyzeResponse,
    WebhookPayload, EOCReportMessage
)
from database import get_db, CallRepository, WebhookRepository
from vapi_fetcher import VAPIClient, fetch_recent_calls, sync_calls_to_db
from analysis_engine import AnalysisEngine, quick_analyze

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Lifespan ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    yield
    logger.info("Shutting down...")


# ============== App Setup ==============
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Fetch and analyze VAPI call data",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Health Check ==============
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "vapi_configured": bool(settings.VAPI_API_KEY),
        "openai_configured": bool(settings.OPENAI_API_KEY)
    }


# ============== Call Fetching Endpoints ==============
@app.get("/api/calls", response_model=FetchCallsResponse, tags=["Calls"])
async def list_calls(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    hours: int = Query(24, ge=1, le=720),
    assistant_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List calls from database
    """
    try:
        start_date = datetime.utcnow() - timedelta(hours=hours)
        
        calls = CallRepository.get_calls(
            db=db,
            limit=limit,
            offset=offset,
            start_date=start_date,
            assistant_id=assistant_id
        )
        
        return FetchCallsResponse(
            success=True,
            total_calls=len(calls),
            calls=calls
        )
    except Exception as e:
        logger.error(f"Error listing calls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calls/{call_id}", response_model=VAPICall, tags=["Calls"])
async def get_call(call_id: str, db: Session = Depends(get_db)):
    """
    Get a specific call by ID
    """
    call = CallRepository.get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@app.post("/api/calls/fetch", response_model=FetchCallsResponse, tags=["Calls"])
async def fetch_calls_from_vapi(
    request: FetchCallsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Fetch calls from VAPI API and store in database
    """
    try:
        async with VAPIClient() as client:
            calls = await client.list_calls(
                limit=request.limit,
                created_at_gt=request.start_date,
                created_at_lt=request.end_date,
                assistant_id=request.assistant_id
            )
        
        # Save to database
        for call in calls:
            CallRepository.save_call(db, call)
        
        return FetchCallsResponse(
            success=True,
            total_calls=len(calls),
            calls=calls
        )
    except Exception as e:
        logger.error(f"Error fetching calls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calls/sync", tags=["Calls"])
async def sync_calls(
    hours: int = Query(24, ge=1, le=168),
    assistant_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Sync calls from VAPI to local database
    """
    try:
        count = await sync_calls_to_db(hours=hours, assistant_id=assistant_id)
        return {
            "success": True,
            "synced_calls": count,
            "hours_synced": hours
        }
    except Exception as e:
        logger.error(f"Error syncing calls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Analysis Endpoints ==============
@app.post("/api/analyze/{call_id}", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_call(
    call_id: str,
    use_llm: bool = Query(False, description="Use LLM for enhanced analysis"),
    db: Session = Depends(get_db)
):
    """
    Analyze a specific call's transcript
    """
    try:
        # Get call from database
        call = CallRepository.get_call(db, call_id)
        
        if not call:
            # Try fetching from VAPI
            async with VAPIClient() as client:
                call = await client.get_call(call_id)
                CallRepository.save_call(db, call)
        
        # Analyze
        engine = AnalysisEngine(use_llm=use_llm)
        
        if use_llm:
            analysis = await engine.analyze_call_async(call)
        else:
            analysis = engine.analyze_call(call)
        
        # Save analysis
        CallRepository.save_analysis(db, call_id, analysis)
        
        return AnalyzeResponse(
            success=True,
            call_id=call_id,
            analysis=analysis
        )
    except Exception as e:
        logger.error(f"Error analyzing call: {e}")
        return AnalyzeResponse(
            success=False,
            call_id=call_id,
            error=str(e)
        )


@app.post("/api/analyze/batch", tags=["Analysis"])
async def analyze_batch(
    call_ids: list[str],
    use_llm: bool = False,
    db: Session = Depends(get_db)
):
    """
    Analyze multiple calls
    """
    results = []
    engine = AnalysisEngine(use_llm=use_llm)
    
    for call_id in call_ids:
        try:
            call = CallRepository.get_call(db, call_id)
            if call:
                if use_llm:
                    analysis = await engine.analyze_call_async(call)
                else:
                    analysis = engine.analyze_call(call)
                CallRepository.save_analysis(db, call_id, analysis)
                results.append({"call_id": call_id, "success": True})
            else:
                results.append({"call_id": call_id, "success": False, "error": "Not found"})
        except Exception as e:
            results.append({"call_id": call_id, "success": False, "error": str(e)})
    
    return {"results": results}


# ============== Statistics Endpoints ==============
@app.get("/api/stats", response_model=CallStats, tags=["Statistics"])
async def get_statistics(
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db)
):
    """
    Get call statistics
    """
    start_date = datetime.utcnow() - timedelta(hours=hours)
    return CallRepository.get_stats(db, start_date=start_date)


@app.get("/api/stats/dashboard", tags=["Statistics"])
async def get_dashboard_stats(
    days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """
    Get dashboard statistics with daily breakdown
    """
    from sqlalchemy import func
    from database import CallRecord
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Overall stats
    stats = CallRepository.get_stats(db, start_date=start_date, end_date=end_date)
    
    # Daily breakdown
    daily_stats = []
    for i in range(days):
        day_start = start_date + timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        day_stats = CallRepository.get_stats(db, start_date=day_start, end_date=day_end)
        daily_stats.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "calls": day_stats.total_calls,
            "duration_minutes": day_stats.total_duration_minutes,
            "cost": day_stats.total_cost
        })
    
    return {
        "overall": stats,
        "daily": daily_stats,
        "period_days": days
    }


# ============== Webhook Endpoints ==============
@app.post("/webhook/vapi", tags=["Webhook"])
async def vapi_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Handle VAPI webhook events (end-of-call reports)
    """
    try:
        payload = await request.json()
        
        # Log webhook
        message = payload.get("message", {})
        event_type = message.get("type", "unknown")
        call_id = message.get("call", {}).get("id") if isinstance(message.get("call"), dict) else None
        
        WebhookRepository.log_webhook(db, event_type, payload, call_id)
        
        # Handle end-of-call report
        if event_type == "end-of-call-report":
            logger.info(f"Received end-of-call report for: {call_id}")
            
            # Process in background
            background_tasks.add_task(
                process_eoc_report,
                payload=payload,
                call_id=call_id
            )
            
            return {"status": "accepted", "call_id": call_id}
        
        return {"status": "ignored", "event_type": event_type}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


async def process_eoc_report(payload: dict, call_id: str):
    """Background task to process end-of-call report"""
    from database import get_db_session
    
    try:
        message = payload.get("message", {})
        call_data = message.get("call", {})
        
        if not call_data:
            logger.warning(f"No call data in EOC report: {call_id}")
            return
        
        # Parse call
        async with VAPIClient() as client:
            call = client._parse_call(call_data)
        
        # Add transcript from message if available
        if message.get("transcript"):
            call.transcript = message["transcript"]
        
        # Save to database
        with get_db_session() as db:
            CallRepository.save_call(db, call)
            
            # Auto-analyze
            engine = AnalysisEngine(use_llm=False)
            analysis = engine.analyze_call(call)
            CallRepository.save_analysis(db, call_id, analysis)
        
        logger.info(f"Processed EOC report: {call_id}")
        
    except Exception as e:
        logger.error(f"Error processing EOC report {call_id}: {e}")


# ============== Assistant Endpoints ==============
@app.get("/api/assistants", tags=["Assistants"])
async def list_assistants():
    """
    List all VAPI assistants
    """
    try:
        async with VAPIClient() as client:
            assistants = await client.list_assistants()
        return {"assistants": assistants}
    except Exception as e:
        logger.error(f"Error listing assistants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/phone-numbers", tags=["Phone Numbers"])
async def list_phone_numbers():
    """
    List all VAPI phone numbers
    """
    try:
        async with VAPIClient() as client:
            numbers = await client.list_phone_numbers()
        return {"phone_numbers": numbers}
    except Exception as e:
        logger.error(f"Error listing phone numbers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== EOC Endpoints ==============
@app.get("/api/vapi/eoc", tags=["EOC"])
async def get_eoc_data(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back for EOC data"),
    assistant_id: Optional[str] = Query(None, description="Filter by assistant ID"),
    db: Session = Depends(get_db)
):
    """
    Fetch End of Call (EOC) data from VAPI
    
    This endpoint fetches ended calls (EOC data) from the VAPI API and returns them
    in a structured JSON format.
    """
    try:
        logger.info(f"EOC: Fetching ended calls for the last {hours} hours.")
        
        # Fetch ended calls from VAPI
        async with VAPIClient() as client:
            since = datetime.utcnow() - timedelta(hours=hours)
            calls = await client.fetch_all_ended_calls(
                since=since,
                assistant_id=assistant_id
            )
        
        logger.info(f"EOC: Fetched {len(calls)} calls from VAPI.")
        
        # Save to database
        saved_count = 0
        for call in calls:
            try:
                CallRepository.save_call(db, call)
                saved_count += 1
            except Exception as e:
                logger.error(f"EOC: Failed to save call {call.id}: {e}")
        
        logger.info(f"EOC: Saved {saved_count} calls to the database.")
        
        # Analyze calls and prepare JSON response
        eoc_data = []
        analysis_engine = AnalysisEngine(use_llm=False)
        
        for call in calls:
            # Perform analysis
            analysis = analysis_engine.analyze_call(call)
            
            # Prepare call data with analysis
            call_data = {
                "call_id": call.id,
                "status": call.status.value if call.status else None,
                "ended_reason": call.ended_reason.value if call.ended_reason else None,
                "created_at": call.created_at.isoformat() if call.created_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                "duration_seconds": call.duration_seconds,
                "cost": call.cost,
                "transcript": call.transcript,
                "recording_url": call.recording_url,
                "metadata": call.metadata.dict() if call.metadata else None,
                "analysis": {
                    "sentiment": {
                        "overall": analysis.sentiment.overall.value,
                        "score": analysis.sentiment.score,
                        "confidence": analysis.sentiment.confidence
                    },
                    "topics": {
                        "topics": analysis.topics.topics,
                        "keywords": analysis.topics.keywords,
                        "entities": analysis.topics.entities
                    },
                    "summary": analysis.summary,
                    "customer_intent": analysis.customer_intent,
                    "resolution_status": analysis.resolution_status,
                    "talk_ratio": analysis.talk_ratio,
                    "word_count": analysis.word_count
                }
            }
            eoc_data.append(call_data)
        
        response_data = {
            "success": True,
            "total_calls": len(calls),
            "hours_looked_back": hours,
            "eoc_data": eoc_data
        }
        
        # Save JSON output to a file
        import json
        with open("eoc_output.json", "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error fetching EOC data: {e}", exc_info=True)
        error_response = {
            "success": False,
            "error": "Failed to fetch EOC data",
            "detail": str(e)
        }
        return JSONResponse(status_code=500, content=error_response)


# ============== Ongoing Calls Endpoint ==============
@app.get("/api/vapi/ongoing-calls", tags=["Ongoing Calls"])
async def get_ongoing_calls(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back for ongoing calls"),
    assistant_id: Optional[str] = Query(None, description="Filter by assistant ID"),
    db: Session = Depends(get_db)
):
    """
    Fetch ongoing calls from VAPI
    
    This endpoint fetches ongoing calls (non-ended calls) from the VAPI API and returns them
    in a structured JSON format.
    """
    try:
        logger.info(f"Ongoing Calls: Fetching calls for the last {hours} hours.")
        
        # Fetch calls from VAPI
        async with VAPIClient() as client:
            since = datetime.utcnow() - timedelta(hours=hours)
            calls = await client.list_calls(
                limit=1000,
                created_at_gt=since,
                assistant_id=assistant_id
            )
        
        logger.info(f"Ongoing Calls: Fetched {len(calls)} calls from VAPI.")
        
        # Filter to only ongoing calls (non-ended)
        ongoing_calls = [c for c in calls if c.status != CallStatus.ENDED]
        
        logger.info(f"Ongoing Calls: Found {len(ongoing_calls)} ongoing calls.")
        
        # Save to database
        saved_count = 0
        for call in ongoing_calls:
            try:
                CallRepository.save_call(db, call)
                saved_count += 1
            except Exception as e:
                logger.error(f"Ongoing Calls: Failed to save call {call.id}: {e}")
        
        logger.info(f"Ongoing Calls: Saved {saved_count} calls to the database.")
        
        # Prepare JSON response
        ongoing_data = [
            {
                "call_id": call.id,
                "status": call.status.value if call.status else None,
                "created_at": call.created_at.isoformat() if call.created_at else None,
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "duration_seconds": call.duration_seconds,
                "cost": call.cost,
                "transcript": call.transcript,
                "recording_url": call.recording_url,
                "metadata": call.metadata.dict() if call.metadata else None
            } for call in ongoing_calls
        ]
        
        response_data = {
            "success": True,
            "total_calls": len(ongoing_calls),
            "hours_looked_back": hours,
            "ongoing_calls": ongoing_data
        }
        
        # Save JSON output to a file
        import json
        with open("ongoing_calls_output.json", "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error fetching ongoing calls: {e}", exc_info=True)
        error_response = {
            "success": False,
            "error": "Failed to fetch ongoing calls",
            "detail": str(e)
        }
        return JSONResponse(status_code=500, content=error_response)


# ============== Export Endpoints ==============
@app.get("/api/export/calls", tags=["Export"])
async def export_calls(
    format: str = Query("json", enum=["json", "csv"]),
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db)
):
    """
    Export calls data
    """
    start_date = datetime.utcnow() - timedelta(hours=hours)
    calls = CallRepository.get_calls(db, limit=10000, start_date=start_date)
    
    if format == "csv":
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "id", "created_at", "duration_seconds", "cost",
                "status", "ended_reason", "transcript"
            ]
        )
        writer.writeheader()
        
        for call in calls:
            writer.writerow({
                "id": call.id,
                "created_at": call.created_at.isoformat() if call.created_at else "",
                "duration_seconds": call.duration_seconds or 0,
                "cost": call.cost or 0,
                "status": call.status.value if call.status else "",
                "ended_reason": call.ended_reason.value if call.ended_reason else "",
                "transcript": (call.transcript or "")[:500]  # Truncate
            })
        
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=calls_export.csv"}
        )
    
    # JSON format
    return {
        "export_date": datetime.utcnow().isoformat(),
        "total_calls": len(calls),
        "calls": [call.dict() for call in calls]
    }


# ============== Error Handler ==============
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# ============== Main Entry Point ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )