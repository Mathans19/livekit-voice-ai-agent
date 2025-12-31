

from flask import Flask, request, jsonify
from flask_cors import CORS 
from livekit import api
import os
from resume_cache import ResumeCache
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
API_KEY = os.getenv("LIVEKIT_API_KEY") 
API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Initialize disk-based resume cache
resume_cache = ResumeCache()

@app.route("/token")
def token():
    """Generate LiveKit access token"""
    identity = request.args.get("identity", "user")
    room = request.args.get("room", "voice")

    grant = api.AccessToken(API_KEY, API_SECRET) \
        .with_identity(identity) \
        .with_grants(api.VideoGrants(room_join=True, room=room))

    return jsonify({
        "url": LIVEKIT_URL,
        "token": grant.to_jwt(),
        "room": room
    })

@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    """Upload and process resume"""
    try:
        data = request.json
        resume_text = data.get("resume", "")
        
        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({
                "success": False,
                "error": "Resume text too short or empty"
            }), 400
        
        # Save to disk cache and generate questions
        result = resume_cache.save_resume(resume_text)
        
        if result['success']:
            return jsonify({
                "success": True,
                "message": f"Resume uploaded! {result['questions_count']} questions generated",
                "questions_count": result['questions_count'],
                "questions": result['questions']
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error')
            }), 500
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/resume-status", methods=["GET"])
def resume_status():
    """Check if resume is cached"""
    has_resume = resume_cache.has_resume()
    metadata = resume_cache.get_metadata()
    
    return jsonify({
        "has_resume": has_resume,
        "metadata": metadata
    })

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Clear resume cache"""
    try:
        resume_cache.clear_cache()
        return jsonify({
            "success": True,
            "message": "Cache cleared"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server on port 3000...")
    print(f"📁 Resume cache directory: {resume_cache.cache_dir.absolute()}")
    app.run(port=3000, debug=True)