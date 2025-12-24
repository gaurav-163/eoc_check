"""
VAPI API client for fetching call data
"""
import httpx
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from config import settings
from models import (
    VAPICall, TranscriptMessage, VAPICallMetadata,
    CostBreakdown, CallStatus, EndedReason
)

logger = logging.getLogger(__name__)


class VAPIError(Exception):
    """VAPI API Error"""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class VAPIClient:
    """
    Async client for VAPI API
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: float = 30.0
    ):
        self.api_key = api_key or settings.VAPI_API_KEY
        self.base_url = (base_url or settings.VAPI_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client:
            await self._client.aclose()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        json_data: Dict = None
    ) -> Dict[str, Any]:
        """Make HTTP request to VAPI API"""
        if not self._client:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout
            )
        
        try:
            response = await self._client.request(
                method=method,
                url=endpoint,
                params=params,
                json=json_data
            )
            
            if response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                return await self._request(method, endpoint, params, json_data)
            
            if response.status_code == 400:
                # Handle bad request errors
                error_data = response.json()
                logger.error(f"VAPI API bad request: {response.status_code} - {error_data}")
                raise VAPIError(
                    message=f"Bad request: {error_data.get('message', 'Invalid request')}",
                    status_code=response.status_code,
                    response=error_data
                )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"VAPI API error: {e.response.status_code} - {e.response.text}")
            raise VAPIError(
                message=f"API request failed: {e.response.text}",
                status_code=e.response.status_code,
                response=e.response.json() if e.response.text else None
            )
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise VAPIError(message=f"Request failed: {str(e)}")
    
    # ============== Call Operations ==============
    async def get_call(self, call_id: str) -> VAPICall:
        """Get a single call by ID"""
        logger.info(f"Fetching call: {call_id}")
        data = await self._request("GET", f"/call/{call_id}")
        return self._parse_call(data)
    
    async def list_calls(
        self,
        limit: int = 100,
        created_at_gt: datetime = None,
        created_at_lt: datetime = None,
        assistant_id: str = None,
        phone_number_id: str = None
    ) -> List[VAPICall]:
        """
        List calls with filters
        
        Args:
            limit: Maximum number of calls to return
            created_at_gt: Filter calls created after this time
            created_at_lt: Filter calls created before this time
            assistant_id: Filter by assistant ID
            phone_number_id: Filter by phone number ID
        """
        logger.info(f"Listing calls (limit={limit})")
        
        params = {"limit": min(limit, 100)}  # VAPI max is 100 per request
        
        if created_at_gt:
            params["createdAtGt"] = created_at_gt.isoformat()
        if created_at_lt:
            params["createdAtLt"] = created_at_lt.isoformat()
        if assistant_id:
            params["assistantId"] = assistant_id
        if phone_number_id:
            params["phoneNumberId"] = phone_number_id
        
        # Handle pagination for large requests
        all_calls = []
        remaining = limit
        
        while remaining > 0:
            params["limit"] = min(remaining, 100)
            data = await self._request("GET", "/call", params=params)
            
            if not data:
                break
            
            calls = [self._parse_call(c) for c in data]
            all_calls.extend(calls)
            
            if len(data) < 100:
                break  # No more pages
            
            remaining -= len(calls)
            
            # Update cursor for next page (use last call's created_at)
            if calls:
                params["createdAtLt"] = calls[-1].created_at.isoformat()
        
        logger.info(f"Fetched {len(all_calls)} calls")
        return all_calls
    
    async def fetch_all_ended_calls(
        self,
        since: datetime = None,
        until: datetime = None,
        assistant_id: str = None
    ) -> List[VAPICall]:
        """
        Fetch all ended calls within a date range
        
        Args:
            since: Start date (default: 24 hours ago)
            until: End date (default: now)
            assistant_id: Filter by assistant
        """
        if not since:
            since = datetime.utcnow() - timedelta(hours=24)
        if not until:
            until = datetime.utcnow()
        
        logger.info(f"Fetching ended calls from {since} to {until}")
        
        calls = await self.list_calls(
            limit=1000,
            created_at_gt=since,
            created_at_lt=until,
            assistant_id=assistant_id
        )
        
        # Filter to only ended calls
        ended_calls = [c for c in calls if c.status == CallStatus.ENDED]
        
        logger.info(f"Found {len(ended_calls)} ended calls")
        return ended_calls
    
    # ============== Assistant Operations ==============
    async def get_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """Get assistant details"""
        return await self._request("GET", f"/assistant/{assistant_id}")
    
    async def list_assistants(self) -> List[Dict[str, Any]]:
        """List all assistants"""
        return await self._request("GET", "/assistant")
    
    # ============== Phone Number Operations ==============
    async def list_phone_numbers(self) -> List[Dict[str, Any]]:
        """List all phone numbers"""
        return await self._request("GET", "/phone-number")
    
    # ============== Parsing Helpers ==============
    def _parse_call(self, data: Dict[str, Any]) -> VAPICall:
        """Parse VAPI API response into VAPICall model"""
        
        # Parse messages
        messages = []
        if "messages" in data and data["messages"]:
            for msg in data["messages"]:
                messages.append(TranscriptMessage(
                    role=msg.get("role", "unknown"),
                    message=msg.get("message", msg.get("content", "")),
                    timestamp=msg.get("time"),
                    duration=msg.get("duration")
                ))
        
        # Parse cost breakdown
        cost_breakdown = None
        if "costBreakdown" in data:
            cb = data["costBreakdown"]
            cost_breakdown = CostBreakdown(
                stt=cb.get("stt", 0),
                llm=cb.get("llm", 0),
                tts=cb.get("tts", 0),
                vapi=cb.get("vapi", 0),
                transport=cb.get("transport", 0),
                total=cb.get("total", data.get("cost", 0))
            )
        
        # Parse metadata
        metadata = VAPICallMetadata(
            assistant_id=data.get("assistantId"),
            phone_number=data.get("phoneNumberId"),
            customer_phone=data.get("customer", {}).get("number") if data.get("customer") else None
        )
        
        # Parse status
        status = CallStatus.ENDED
        if data.get("status"):
            try:
                status = CallStatus(data["status"])
            except ValueError:
                pass
        
        # Parse ended reason
        ended_reason = None
        if data.get("endedReason"):
            try:
                ended_reason = EndedReason(data["endedReason"])
            except ValueError:
                pass
        
        # Parse timestamps
        def parse_dt(val):
            if not val:
                return None
            if isinstance(val, datetime):
                return val
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except:
                return None
        
        return VAPICall(
            id=data["id"],
            org_id=data.get("orgId"),
            type=data.get("type", "webCall"),
            status=status,
            ended_reason=ended_reason,
            created_at=parse_dt(data.get("createdAt")) or datetime.utcnow(),
            started_at=parse_dt(data.get("startedAt")),
            ended_at=parse_dt(data.get("endedAt")),
            duration_seconds=data.get("durationSeconds", data.get("duration", 0)),
            transcript=data.get("transcript"),
            messages=messages,
            recording_url=data.get("recordingUrl"),
            stereo_recording_url=data.get("stereoRecordingUrl"),
            cost=data.get("cost"),
            cost_breakdown=cost_breakdown,
            metadata=metadata,
            raw_data=data
        )


# ============== Convenience Functions ==============
async def fetch_recent_calls(hours: int = 24) -> List[VAPICall]:
    """Fetch calls from the last N hours"""
    async with VAPIClient() as client:
        since = datetime.utcnow() - timedelta(hours=hours)
        return await client.fetch_all_ended_calls(since=since)


async def fetch_call_by_id(call_id: str) -> VAPICall:
    """Fetch a single call by ID"""
    async with VAPIClient() as client:
        return await client.get_call(call_id)


async def sync_calls_to_db(
    hours: int = 24,
    assistant_id: str = None
) -> int:
    """Fetch calls and save to database"""
    from database import get_db_session, CallRepository
    
    calls = []
    async with VAPIClient() as client:
        since = datetime.utcnow() - timedelta(hours=hours)
        calls = await client.fetch_all_ended_calls(
            since=since,
            assistant_id=assistant_id
        )
    
    saved = 0
    with get_db_session() as db:
        for call in calls:
            CallRepository.save_call(db, call)
            saved += 1
    
    logger.info(f"Synced {saved} calls to database")
    return saved