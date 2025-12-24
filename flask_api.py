"""
Flask API to expose Python scripts for n8n integration
"""
from flask import Flask, jsonify, request
from analysis_engine import AnalysisEngine, quick_analyze
from vapi_fetcher import VAPIClient, fetch_recent_calls, fetch_call_by_id
from models import VAPICall
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Analysis Engine
analysis_engine = AnalysisEngine(use_llm=False)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Flask API for n8n integration is running"
    })


@app.route('/api/analyze', methods=['POST'])
async def analyze_transcript():
    """Analyze a call transcript"""
    try:
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON"
            }), 400
        
        data = request.get_json()
        call_id = data.get('call_id')
        
        if not call_id:
            return jsonify({
                "error": "call_id is required"
            }), 400
        
        # Validate call_id format (UUID)
        import re
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        if not uuid_pattern.match(call_id):
            return jsonify({
                "error": "call_id must be a valid UUID"
            }), 400
        
        # Fetch call data
        try:
            call = await fetch_call_by_id(call_id)
        except Exception as fetch_error:
            logger.error(f"Error fetching call data: {fetch_error}")
            return jsonify({
                "error": "Failed to fetch call data. Please check the call_id and try again."
            }), 404
        
        # Perform analysis
        try:
            analysis = quick_analyze(call)
        except Exception as analysis_error:
            logger.error(f"Error analyzing call: {analysis_error}")
            return jsonify({
                "error": "Failed to analyze call. Please try again later."
            }), 500
        
        return jsonify({
            "success": True,
            "call_id": call_id,
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
        })
    except Exception as e:
        logger.error(f"Unexpected error analyzing transcript: {e}")
        return jsonify({
            "error": "An unexpected error occurred. Please try again later."
        }), 500


@app.route('/api/analyze-batch', methods=['POST'])
async def analyze_batch():
    """Analyze multiple call transcripts in batch"""
    try:
        data = request.get_json()
        call_ids = data.get('call_ids', [])
        batch_size = data.get('batch_size', 1000)
        max_concurrency = data.get('max_concurrency', 10)
        
        if not call_ids:
            return jsonify({
                "error": "call_ids is required"
            }), 400
        
        # Fetch calls
        calls = []
        async with VAPIClient() as client:
            for call_id in call_ids:
                call = await client.get_call(call_id)
                calls.append(call)
        
        # Perform batch analysis
        from analysis_engine import AnalysisEngine
        engine = AnalysisEngine(use_llm=False)
        analyses = await engine.analyze_calls_batch_async(
            calls=calls,
            use_llm=False,
            batch_size=batch_size,
            max_concurrency=max_concurrency
        )
        
        # Prepare response
        results = []
        for call_id, analysis in zip(call_ids, analyses):
            results.append({
                "call_id": call_id,
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
            })
        
        return jsonify({
            "success": True,
            "total_calls": len(results),
            "results": results
        })
    except Exception as e:
        logger.error(f"Error analyzing batch: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/fetch-calls', methods=['GET'])
async def fetch_calls():
    """Fetch recent calls from VAPI"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        calls = await fetch_recent_calls(hours=hours)
        
        calls_data = []
        for call in calls:
            calls_data.append({
                "call_id": call.id,
                "status": call.status.value if call.status else None,
                "created_at": call.created_at.isoformat() if call.created_at else None,
                "duration_seconds": call.duration_seconds,
                "cost": call.cost
            })
        
        return jsonify({
            "success": True,
            "total_calls": len(calls_data),
            "calls": calls_data
        })
    except Exception as e:
        logger.error(f"Error fetching calls: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/call-details/<call_id>', methods=['GET'])
async def get_call_details(call_id: str):
    """Get detailed information for a specific call"""
    try:
        call = await fetch_call_by_id(call_id)
        
        return jsonify({
            "success": True,
            "call_id": call.id,
            "status": call.status.value if call.status else None,
            "created_at": call.created_at.isoformat() if call.created_at else None,
            "ended_at": call.ended_at.isoformat() if call.ended_at else None,
            "duration_seconds": call.duration_seconds,
            "cost": call.cost,
            "transcript": call.transcript,
            "recording_url": call.recording_url,
            "metadata": call.metadata.dict() if call.metadata else None
        })
    except Exception as e:
        logger.error(f"Error fetching call details: {e}")
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)