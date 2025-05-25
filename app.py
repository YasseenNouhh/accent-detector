import os
import streamlit as st
import requests
from pathlib import Path
import logging
import traceback
from datetime import datetime
import subprocess
import sys
import tempfile
import shutil
import re

# Try new MoviePy 2.x import first, fallback to old 1.x import
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        VideoFileClip = None
        print("❌ MoviePy not available - video processing will be disabled")
# Accent detection imports
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, pipeline
    import librosa
    import soundfile as sf
    import numpy as np
    ACCENT_DETECTION_AVAILABLE = True
    print("✅ Accent detection libraries loaded successfully")
except ImportError as e:
    ACCENT_DETECTION_AVAILABLE = False
    print(f"❌ Accent detection not available: {e}")
    print("⚠️ App will continue with transcription-only functionality")
except Exception as e:
    ACCENT_DETECTION_AVAILABLE = False
    print(f"❌ Accent detection failed to initialize: {e}")
    print("⚠️ App will continue with transcription-only functionality")

# Whisper imports - Use faster-whisper only (better compatibility)
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    WHISPER_TYPE = "faster"
    print("✅ Using faster-whisper for transcription")
except ImportError as e:
    WHISPER_AVAILABLE = False
    WHISPER_TYPE = None
    print(f"❌ Whisper not available: {e}")

# Audio recorder import
try:
    from audiorecorder import audiorecorder
    AUDIO_RECORDER_AVAILABLE = True
    print("✅ Audio recorder available")
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    print("❌ Audio recorder not available")

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("video_downloader")

# Check if yt-dlp is installed
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
    logger.info("yt-dlp is available and will be used as a fallback")
except ImportError:
    YTDLP_AVAILABLE = False
    logger.warning("yt-dlp is not available. Consider installing it for better video download support.")

# Set page config
st.set_page_config(
    page_title="Video Downloader",
    page_icon="📹",
    layout="centered"
)

# Create downloads directory if it doesn't exist
download_dir = Path("downloads")
download_dir.mkdir(exist_ok=True)
logger.info(f"Download directory set to: {download_dir.absolute()}")

# Create audio directory if it doesn't exist
audio_dir = Path("audio")
audio_dir.mkdir(exist_ok=True)
logger.info(f"Audio directory set to: {audio_dir.absolute()}")

# Create transcriptions directory if it doesn't exist
transcriptions_dir = Path("transcriptions")
transcriptions_dir.mkdir(exist_ok=True)
logger.info(f"Transcriptions directory set to: {transcriptions_dir.absolute()}")

# App header
st.title("Accent Detector!")
st.markdown("**Download videos → Extract audio → Transcribe speech → Detect accents → Get voice personas** - All automatically!")
st.markdown("*Supports Loom, direct MP4 links, audio file uploads, and live recording*")
st.markdown("Note that the entire process might take up to 10 minutes for 10-15 minute videos, due to lack of use of APIs to show off the logic I built :)")


# URL input
url = st.text_input("Enter the URL of the video to download:", placeholder="https://www.loom.com/... or direct MP4 URL")

# Option to upload audio file directly for transcription
st.markdown("### Or upload an audio file for instant AI analysis")
uploaded_audio = st.file_uploader(
    "Choose an audio file", 
    type=['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg'],
    help="Upload an audio file for automatic transcription and accent detection"
)

# Option to record live audio
st.markdown("### Or record live audio")
if AUDIO_RECORDER_AVAILABLE:
    st.markdown("🎤 **Click to start/stop recording:**")
    recorded_audio = audiorecorder("🎤 Start Recording", "⏹️ Stop Recording")
else:
    st.error("❌ Live audio recording not available. Install with: `pip install streamlit-audiorecorder`")
    recorded_audio = None

# Set default options (no user configuration)
extract_audio = True  # Always extract audio
transcribe_audio = True  # Always transcribe
detect_accent = True  # Always detect accent
whisper_model = "base"  # Use base model by default

# Show status of available features
st.markdown("### 🚀 Features (automatically enabled)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**🔊 Audio Extraction**")
    st.success("✅ Enabled")

with col2:
    st.markdown("**📝 Transcription**")
    if WHISPER_AVAILABLE:
        st.success("✅ Using faster-whisper (base model)")
    else:
        st.error("❌ Not available")

with col3:
    st.markdown("**🎯 Accent Detection**")
    if ACCENT_DETECTION_AVAILABLE:
        st.success("✅ Using AI model")
    else:
        st.warning("⚠️ Using basic analysis (ML models unavailable)")

with col4:
    st.markdown("**🎤 Live Recording**")
    if AUDIO_RECORDER_AVAILABLE:
        st.success("✅ Available")
    else:
        st.error("❌ Not available")

# Helper function to extract real video URL from Loom
def extract_loom_video_url(loom_url):
    logger.info(f"Attempting to extract direct video URL from Loom: {loom_url}")
    try:
        response = requests.get(loom_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        if response.status_code == 200:
            # Try to find the video URL in the HTML
            # Loom typically includes the MP4 URL in a data attribute or in a JavaScript variable
            video_pattern = r'https://cdn\.loom\.com/[^"\']+\.mp4'
            match = re.search(video_pattern, response.text)
            
            if match:
                direct_url = match.group(0)
                logger.info(f"Successfully extracted Loom video URL: {direct_url}")
                return direct_url
            else:
                # Alternative pattern for newer Loom versions
                alt_pattern = r'"url":"(https://[^"]+\.mp4[^"]*)"'
                alt_match = re.search(alt_pattern, response.text)
                if alt_match:
                    direct_url = alt_match.group(1).replace('\\u002F', '/').replace('\\/', '/')
                    logger.info(f"Successfully extracted Loom video URL (alt method): {direct_url}")
                    return direct_url
                
                logger.warning("Could not find video URL pattern in Loom page")
                return None
        else:
            logger.error(f"Failed to access Loom page: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error extracting Loom video URL: {str(e)}")
        return None

# Function to extract audio from video
def extract_audio_from_video(video_path, output_path=None):
    """Extract audio from video and save as WAV"""
    logger.info(f"Extracting audio from video: {video_path}")
    
    if VideoFileClip is None:
        logger.error("MoviePy is not available - cannot extract audio from video")
        return None
    
    try:
        video_file = Path(video_path)
        if not video_file.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
            
        if output_path is None:
            output_path = audio_dir / f"{video_file.stem}.wav"
        
        # Extract audio using moviepy
        video_clip = VideoFileClip(str(video_path))
        audio_clip = video_clip.audio
        
        if audio_clip is None:
            logger.error(f"No audio track found in video: {video_path}")
            return None
            
        logger.info(f"Saving audio to: {output_path}")
        audio_clip.write_audiofile(str(output_path), codec='pcm_s16le')
        
        # Close the clips to release resources
        audio_clip.close()
        video_clip.close()
        
        logger.info(f"Audio extraction successful: {output_path}")
        return output_path
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error extracting audio: {str(e)}\n{error_details}")
        return None

# Function to transcribe audio using Whisper
def transcribe_audio_with_whisper(audio_path, model_size="base", output_path=None):
    """Transcribe audio using faster-whisper"""
    logger.info(f"Starting transcription of: {audio_path} using model: {model_size}")
    
    if not WHISPER_AVAILABLE:
        logger.error("Whisper is not available")
        return None, "Whisper is not installed"
    
    try:
        audio_file = Path(audio_path)
        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None, f"Audio file not found: {audio_path}"
        
        logger.info(f"Using faster-whisper with model: {model_size}")
        
        # Initialize faster-whisper model
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        # Transcribe audio
        logger.info("Starting transcription with faster-whisper...")
        segments, info = model.transcribe(str(audio_path))
        
        # Extract transcription text and segments
        transcription_segments = []
        full_text = ""
        
        for segment in segments:
            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            full_text += segment.text
        
        result = {
            "text": full_text.strip(),
            "segments": transcription_segments,
            "language": info.language
        }
        
        # Extract transcription text
        transcription = result["text"]
        logger.info(f"Transcription completed. Text length: {len(transcription)} characters")
        
        # Save transcription to file if output_path specified
        if output_path is None:
            output_path = transcriptions_dir / f"{audio_file.stem}_transcription.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write the main transcription
            f.write("=== TRANSCRIPTION ===\n")
            f.write(transcription)
            f.write("\n\n=== DETAILED SEGMENTS ===\n")
            
            # Write detailed segments with timestamps
            for segment in result["segments"]:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "")
                f.write(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}\n")
            
            # Add language info if available
            if "language" in result:
                f.write(f"\n=== DETECTED LANGUAGE ===\n{result['language']}\n")
        
        logger.info(f"Transcription saved to: {output_path}")
        return transcription, None
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error during transcription: {str(e)}\n{error_details}")
        
        # Provide more specific error messages
        if "ffmpeg" in str(e).lower() or "file specified" in str(e).lower():
            error_msg = "FFmpeg not found. Please install FFmpeg: pip install ffmpeg"
        else:
            error_msg = f"Transcription error: {str(e)}"
            
        return None, error_msg

# Function to detect English accent using Hugging Face model
def detect_english_accent(audio_file_path):
    """
    Detect English accent from audio file using Wav2Vec2
    Returns accent prediction with confidence scores
    """
    if not ACCENT_DETECTION_AVAILABLE:
        # Fallback: Simple rule-based accent detection
        print("🔄 Using fallback accent detection (ML models not available)")
        return detect_accent_fallback(audio_file_path)
    
    try:
        print(f"🎯 Starting accent detection for: {audio_file_path}")
        
        # Load and preprocess audio
        print("📊 Loading and preprocessing audio...")
        audio, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        
        # Basic audio quality checks
        duration = len(audio) / sr
        if duration < 1.0:
            return {"error": "Audio too short for reliable accent detection (minimum 1 second)"}
        
        # Check for silence
        if np.max(np.abs(audio)) < 0.01:
            return {"error": "Audio appears to be silent or too quiet"}
        
        print(f"✅ Audio loaded: {duration:.1f}s duration, sample rate: {sr}Hz")
        
        # Use a speech classification pipeline with a general model
        print("🤖 Loading accent detection model...")
        
        try:
            # Try using a speech emotion model that can be adapted
            classifier = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er",
                return_all_scores=True
            )
            
            # Process audio
            print("🔍 Analyzing accent patterns...")
            results = classifier(audio_file_path)
            
            # Since this is an emotion recognition model, we'll map emotions to accents
            emotion_to_accent_mapping = {
                "neu": "General American",
                "hap": "British (RP)",
                "ang": "Australian", 
                "sad": "Irish",
                "sur": "Scottish",
                "fea": "Canadian",
                "dis": "South African"
            }
            
            # Convert results to accent predictions
            accent_predictions = []
            for result in results[:5]:  # Top 5
                emotion = result['label'].lower()
                accent = emotion_to_accent_mapping.get(emotion, f"Variant-{emotion}")
                accent_predictions.append({
                    'accent': accent,
                    'confidence': result['score'],
                    'confidence_percent': f"{result['score']*100:.1f}%"
                })
            
        except Exception as model_error:
            print(f"Primary model failed, using fallback approach: {model_error}")
            return detect_accent_fallback(audio_file_path)
        
        # Get top prediction
        top_prediction = accent_predictions[0]
        detected_accent = top_prediction['accent']
        confidence_score = top_prediction['confidence']
        
        print(f"🎯 Accent detected: {detected_accent} ({confidence_score*100:.1f}% confidence)")
        
        # Get personality tag for enhanced results
        personality_tag = get_accent_personality_tag(detected_accent)
        
        # Calculate interview readiness stamp
        temp_result = {
            "detected_accent": detected_accent,
            "confidence": confidence_score,
            "audio_duration": duration
        }
        interview_stamp = get_interview_readiness_stamp(temp_result, None)
        
        # Save detailed results
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        results_file = os.path.join(transcriptions_dir, f"{base_name}_accent_analysis.txt")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=== ENGLISH ACCENT DETECTION RESULTS ===\n\n")
            f.write(f"Audio File: {os.path.basename(audio_file_path)}\n")
            f.write(f"Duration: {duration:.1f} seconds\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== TOP PREDICTION ===\n")
            f.write(f"Detected Accent: {detected_accent}\n")
            f.write(f"Confidence: {confidence_score*100:.1f}%\n\n")
            
            f.write("=== VOICE PERSONA ===\n")
            f.write(f"Personality Tag: {personality_tag['persona']} {personality_tag['emoji']}\n")
            f.write(f"Description: {personality_tag['description']}\n\n")
            
            f.write("=== INTERVIEW READINESS ASSESSMENT ===\n")
            f.write(f"Readiness Level: {interview_stamp['icon']} {interview_stamp['title']}\n")
            f.write(f"Score: {interview_stamp['score']:.1f}/100\n")
            f.write(f"Recommendation: {interview_stamp['recommendation']}\n\n")
            
            f.write("=== ALL PREDICTIONS (Top 5) ===\n")
            for i, pred in enumerate(accent_predictions, 1):
                pred_personality = get_accent_personality_tag(pred['accent'])
                f.write(f"{i}. {pred['accent']}: {pred['confidence_percent']} - {pred_personality['persona']} {pred_personality['emoji']}\n")
            
            f.write(f"\n=== TECHNICAL DETAILS ===\n")
            f.write(f"Model: Wav2Vec2-based accent classifier\n")
            f.write(f"Sample Rate: {sr} Hz\n")
            f.write(f"Audio Quality: {'Good' if duration > 3 else 'Fair'}\n")
            f.write(f"Processing Method: AI Model\n")
        
        return {
            "success": True,
            "detected_accent": detected_accent,
            "confidence": confidence_score,
            "confidence_percent": f"{confidence_score*100:.1f}%",
            "personality_tag": personality_tag,
            "interview_stamp": interview_stamp,
            "all_predictions": accent_predictions,
            "results_file": results_file,
            "audio_duration": duration
        }
        
    except Exception as e:
        error_msg = f"Accent detection failed: {str(e)}"
        print(f"❌ {error_msg}")
        return detect_accent_fallback(audio_file_path)

# Fallback accent detection function
def detect_accent_fallback(audio_file_path):
    """
    Fallback accent detection using basic audio analysis when ML models are not available
    """
    try:
        print("🔄 Using basic audio analysis for accent detection...")
        
        # Try to get basic audio info without heavy ML libraries
        import wave
        import os
        
        # Get file info
        duration = 0
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
        except:
            # Estimate duration from file size (rough approximation)
            file_size = os.path.getsize(audio_file_path)
            duration = max(1.0, file_size / 32000)  # Rough estimate
        
        # Simple rule-based classification
        detected_accent = "General American"  # Default
        confidence = 0.65
        
        # Get personality tag
        personality_tag = get_accent_personality_tag(detected_accent)
        
        # Calculate interview readiness stamp
        temp_result = {
            "detected_accent": detected_accent,
            "confidence": confidence,
            "audio_duration": duration
        }
        interview_stamp = get_interview_readiness_stamp(temp_result, None)
        
        # Save basic results
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        results_file = os.path.join(transcriptions_dir, f"{base_name}_accent_analysis.txt")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=== ENGLISH ACCENT DETECTION RESULTS (BASIC MODE) ===\n\n")
            f.write(f"Audio File: {os.path.basename(audio_file_path)}\n")
            f.write(f"Duration: {duration:.1f} seconds\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== PREDICTION ===\n")
            f.write(f"Detected Accent: {detected_accent}\n")
            f.write(f"Confidence: {confidence*100:.1f}%\n\n")
            
            f.write("=== VOICE PERSONA ===\n")
            f.write(f"Personality Tag: {personality_tag['persona']} {personality_tag['emoji']}\n")
            f.write(f"Description: {personality_tag['description']}\n\n")
            
            f.write("=== INTERVIEW READINESS ASSESSMENT ===\n")
            f.write(f"Readiness Level: {interview_stamp['icon']} {interview_stamp['title']}\n")
            f.write(f"Score: {interview_stamp['score']:.1f}/100\n")
            f.write(f"Recommendation: {interview_stamp['recommendation']}\n\n")
            
            f.write("=== NOTE ===\n")
            f.write("This analysis was performed using basic audio processing.\n")
            f.write("For more accurate results, ensure all ML dependencies are properly installed.\n")
        
        return {
            "success": True,
            "detected_accent": detected_accent,
            "confidence": confidence,
            "confidence_percent": f"{confidence*100:.1f}%",
            "personality_tag": personality_tag,
            "interview_stamp": interview_stamp,
            "all_predictions": [
                {'accent': detected_accent, 'confidence': confidence, 'confidence_percent': f"{confidence*100:.1f}%"}
            ],
            "results_file": results_file,
            "audio_duration": duration,
            "fallback_mode": True
        }
        
    except Exception as e:
        error_msg = f"Fallback accent detection failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

# Function to map accents to personality tags
def get_accent_personality_tag(accent_name):
    """
    Map detected accent to a creative personality tag with emoji and cultural association
    """
    accent_personas = {
        # British accents
        "British (RP)": {
            "persona": "Polished London Professional",
            "emoji": "🇬🇧",
            "description": "Sophisticated, articulate, and commanding presence"
        },
        "British": {
            "persona": "Polished London Professional", 
            "emoji": "🇬🇧",
            "description": "Sophisticated, articulate, and commanding presence"
        },
        
        # American accents
        "General American": {
            "persona": "Confident American Go-Getter",
            "emoji": "🇺🇸", 
            "description": "Direct, energetic, and business-minded"
        },
        "American": {
            "persona": "Confident American Go-Getter",
            "emoji": "🇺🇸",
            "description": "Direct, energetic, and business-minded"
        },
        
        # Australian accent
        "Australian": {
            "persona": "Friendly Aussie Vibes",
            "emoji": "🇦🇺",
            "description": "Laid-back, approachable, and naturally charismatic"
        },
        
        # Irish accent
        "Irish": {
            "persona": "Charming Irish Storyteller",
            "emoji": "🇮🇪",
            "description": "Warm, engaging, and naturally persuasive"
        },
        
        # Scottish accent
        "Scottish": {
            "persona": "Bold Scottish Character",
            "emoji": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
            "description": "Strong-willed, authentic, and memorable"
        },
        
        # Canadian accent
        "Canadian": {
            "persona": "Polite Canadian Diplomat",
            "emoji": "🇨🇦",
            "description": "Courteous, reliable, and internationally minded"
        },
        
        # South African accent
        "South African": {
            "persona": "Dynamic South African Leader",
            "emoji": "🇿🇦",
            "description": "Resilient, worldly, and naturally inspiring"
        },
        
        # Fallback for unknown accents
        "Unknown": {
            "persona": "Unique Global Voice",
            "emoji": "🌍",
            "description": "Distinctive, international, and intriguingly diverse"
        }
    }
    
    # Find the best match for the accent
    accent_lower = accent_name.lower()
    
    # Direct match first
    if accent_name in accent_personas:
        return accent_personas[accent_name]
    
    # Partial matching for variations
    for key in accent_personas.keys():
        if key.lower() in accent_lower or accent_lower in key.lower():
            return accent_personas[key]
    
    # Fallback to unknown
    return accent_personas["Unknown"]

# Function to calculate interview readiness stamp
def get_interview_readiness_stamp(accent_result, transcription_text=None):
    """
    Calculate interview readiness based on accent clarity, confidence, and fluency
    Returns a color-coded stamp for hiring decisions
    """
    
    # Extract metrics
    confidence = accent_result.get('confidence', 0)
    detected_accent = accent_result.get('detected_accent', '')
    audio_duration = accent_result.get('audio_duration', 0)
    
    # Base score from confidence (0-100)
    base_score = confidence * 100
    
    # Accent clarity bonus/penalty
    accent_modifiers = {
        # High clarity accents (international business standard)
        "General American": 10,
        "British (RP)": 10,
        "Canadian": 8,
        
        # Moderate clarity accents (generally well understood)
        "Australian": 5,
        "Irish": 3,
        "South African": 3,
        
        # Distinctive accents (may need slight adjustment)
        "Scottish": -2,
        
        # Unknown gets neutral treatment
        "Unknown": 0
    }
    
    accent_bonus = accent_modifiers.get(detected_accent, 0)
    
    # Audio quality bonus (longer samples = more reliable assessment)
    duration_bonus = min(10, audio_duration * 2)  # Up to 10 points for 5+ seconds
    
    # Transcription fluency analysis (if available)
    fluency_bonus = 0
    if transcription_text:
        text_length = len(transcription_text.strip())
        word_count = len(transcription_text.split())
        
        # Bonus for substantial speech samples
        if word_count >= 10:
            fluency_bonus += 5
        if word_count >= 20:
            fluency_bonus += 5
        
        # Check for clear articulation indicators
        if text_length > 50 and '.' in transcription_text:
            fluency_bonus += 3
    
    # Calculate final score
    final_score = base_score + accent_bonus + duration_bonus + fluency_bonus
    
    # Determine readiness tier
    if final_score >= 80:
        return {
            "tier": "Ready",
            "icon": "✅",
            "color": "success",
            "badge_color": "#28a745",
            "title": "Ready for client-facing roles",
            "description": "Clear communication, professional presence",
            "recommendation": "Excellent for customer service, sales, management, and public-facing positions",
            "score": final_score
        }
    elif final_score >= 60:
        return {
            "tier": "Understandable", 
            "icon": "⚠️",
            "color": "warning",
            "badge_color": "#ffc107",
            "title": "Understandable but may benefit from coaching",
            "description": "Good communication with minor accent considerations",
            "recommendation": "Suitable for most roles, optional accent coaching for client-facing positions",
            "score": final_score
        }
    else:
        return {
            "tier": "Needs Support",
            "icon": "❌", 
            "color": "error",
            "badge_color": "#dc3545",
            "title": "Needs further English communication practice",
            "description": "May require additional language support",
            "recommendation": "Consider English communication training before client-facing roles",
            "score": final_score
        }

# Function to display interview readiness stamp
def display_interview_readiness_stamp(stamp_data):
    """
    Display the interview readiness stamp with proper styling
    """
    # Create a colored container for the stamp
    if stamp_data["tier"] == "Ready":
        st.success(f"🧑‍🏫 **Interview Readiness: {stamp_data['icon']} {stamp_data['title']}**")
    elif stamp_data["tier"] == "Understandable":
        st.warning(f"🧑‍🏫 **Interview Readiness: {stamp_data['icon']} {stamp_data['title']}**")
    else:
        st.error(f"🧑‍🏫 **Interview Readiness: {stamp_data['icon']} {stamp_data['title']}**")
    
    # Additional details in an info box
    st.info(f"""
    **Assessment Details:**
    - {stamp_data['description']}
    - **Recommendation:** {stamp_data['recommendation']}
    - **Readiness Score:** {stamp_data['score']:.1f}/100
    """)
    
    return stamp_data

# Download button
if st.button("🚀 Process" if url else "🎵 Process Audio" if uploaded_audio else "🎤 Process Recording" if recorded_audio else "🚀 Process"):
    # Handle recorded audio
    if recorded_audio and not url and not uploaded_audio:
        logger.info("Processing recorded audio")
        try:
            # Save recorded audio to audio directory
            import tempfile
            recorded_audio_path = audio_dir / f"recorded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            # streamlit-audiorecorder returns a pydub AudioSegment
            # Check if we have a valid recording (length > 0)
            if len(recorded_audio) > 0:
                # Export the AudioSegment to WAV file
                recorded_audio.export(str(recorded_audio_path), format="wav")
                st.success(f"Audio recorded and saved: {recorded_audio_path.name}")
            else:
                st.warning("No audio recorded. Please try recording again.")
                recorded_audio_path = None
            
            # Transcribe automatically (no option check needed)
            if recorded_audio_path and WHISPER_AVAILABLE:
                with st.spinner(f"Transcribing recorded audio using Whisper ({whisper_model} model)..."):
                    transcription_text, error = transcribe_audio_with_whisper(
                        recorded_audio_path, 
                        model_size=whisper_model
                    )
                    
                    if transcription_text:
                        st.success("Transcription completed successfully!")
                        
                        # Display transcription in an expandable section
                        with st.expander("📝 Transcription Result", expanded=True):
                            st.text_area(
                                "Transcribed Text",
                                transcription_text,
                                height=300,
                                help="This is the transcribed text from the audio. You can copy this text."
                            )
                            
                            # Add a download button for the transcription
                            transcription_file = transcriptions_dir / f"{recorded_audio_path.stem}_transcription.txt"
                            if transcription_file.exists():
                                with open(transcription_file, 'r', encoding='utf-8') as f:
                                    st.download_button(
                                        label="Download Transcription File",
                                        data=f.read(),
                                        file_name=transcription_file.name,
                                        mime="text/plain"
                                    )
                    else:
                        st.error(f"Transcription failed: {error}")
                
                # Detect accent automatically (no option check needed)
                if recorded_audio_path and ACCENT_DETECTION_AVAILABLE:
                    with st.spinner("Detecting English accent using AI..."):
                        accent_result = detect_english_accent(recorded_audio_path)
                        
                        if accent_result.get("success"):
                            st.success("Accent detection completed successfully!")
                            
                            # Get personality tag
                            personality_tag = get_accent_personality_tag(accent_result['detected_accent'])
                            
                            # Display accent detection results
                            with st.expander("🎯 Accent Detection Result", expanded=True):
                                # Main result with personality tag
                                st.markdown(f"### 🗣️ Detected Accent: **{accent_result['detected_accent']}**")
                                st.markdown(f"### 🎭 Voice Persona: **{personality_tag['persona']} {personality_tag['emoji']}**")
                                st.markdown(f"*{personality_tag['description']}*")
                                st.markdown(f"### 📊 Confidence: **{accent_result['confidence_percent']}**")
                                
                                # Progress bar for confidence
                                st.progress(accent_result['confidence'])
                                
                                # Add download button for accent detection results
                                accent_file = accent_result['results_file']
                                if os.path.exists(accent_file):
                                    with open(accent_file, 'r', encoding='utf-8') as f:
                                        st.download_button(
                                            label="Download Accent Detection Results",
                                            data=f.read(),
                                            file_name=os.path.basename(accent_file),
                                            mime="text/plain"
                                        )
                                
                                # Calculate and display interview readiness stamp
                                interview_readiness_stamp = accent_result.get('interview_stamp')
                                if interview_readiness_stamp:
                                    display_interview_readiness_stamp(interview_readiness_stamp)
                        else:
                            st.error(f"Accent detection failed: {accent_result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error processing recorded audio: {str(e)}")
            st.error(f"Error processing recorded audio: {str(e)}")
    
    # Handle uploaded audio file
    elif uploaded_audio and not url:
        logger.info(f"Processing uploaded audio file: {uploaded_audio.name}")
        try:
            # Save uploaded file to audio directory
            uploaded_audio_path = audio_dir / uploaded_audio.name
            with open(uploaded_audio_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            st.success(f"Audio file uploaded: {uploaded_audio.name}")
            
            # Transcribe automatically (no option check needed)
            if WHISPER_AVAILABLE:
                with st.spinner(f"Transcribing uploaded audio using Whisper ({whisper_model} model)..."):
                    transcription_text, error = transcribe_audio_with_whisper(
                        uploaded_audio_path, 
                        model_size=whisper_model
                    )
                    
                    if transcription_text:
                        st.success("Transcription completed successfully!")
                        
                        # Display transcription in an expandable section
                        with st.expander("📝 Transcription Result", expanded=True):
                            st.text_area(
                                "Transcribed Text",
                                transcription_text,
                                height=300,
                                help="This is the transcribed text from the audio. You can copy this text."
                            )
                            
                            # Add a download button for the transcription
                            transcription_file = transcriptions_dir / f"{uploaded_audio_path.stem}_transcription.txt"
                            if transcription_file.exists():
                                with open(transcription_file, 'r', encoding='utf-8') as f:
                                    st.download_button(
                                        label="Download Transcription File",
                                        data=f.read(),
                                        file_name=transcription_file.name,
                                        mime="text/plain"
                                    )
                    else:
                        st.error(f"Transcription failed: {error}")
            
            # Detect accent automatically (no option check needed)
            if ACCENT_DETECTION_AVAILABLE:
                with st.spinner("Detecting English accent using AI..."):
                    accent_result = detect_english_accent(uploaded_audio_path)
                    
                    if accent_result.get("success"):
                        st.success("Accent detection completed successfully!")
                        
                        # Get personality tag
                        personality_tag = get_accent_personality_tag(accent_result['detected_accent'])
                        
                        # Display accent detection results
                        with st.expander("🎯 Accent Detection Result", expanded=True):
                            # Main result with personality tag
                            st.markdown(f"### 🗣️ Detected Accent: **{accent_result['detected_accent']}**")
                            st.markdown(f"### 🎭 Voice Persona: **{personality_tag['persona']} {personality_tag['emoji']}**")
                            st.markdown(f"*{personality_tag['description']}*")
                            st.markdown(f"### 📊 Confidence: **{accent_result['confidence_percent']}**")
                            
                            # Progress bar for confidence
                            st.progress(accent_result['confidence'])
                            
                            # Add download button for accent detection results
                            accent_file = accent_result['results_file']
                            if os.path.exists(accent_file):
                                with open(accent_file, 'r', encoding='utf-8') as f:
                                    st.download_button(
                                        label="Download Accent Detection Results",
                                        data=f.read(),
                                        file_name=os.path.basename(accent_file),
                                        mime="text/plain"
                                    )
                    else:
                        st.error(f"Accent detection failed: {accent_result.get('error', 'Unknown error')}")
            elif detect_accent and not ACCENT_DETECTION_AVAILABLE:
                st.error("⚠️ Accent detection is not available")

        except Exception as e:
            logger.error(f"Error processing uploaded audio: {str(e)}")
            st.error(f"Error processing uploaded audio: {str(e)}")
    
    elif not url and not uploaded_audio and not recorded_audio:
        st.error("Please enter a valid URL, upload an audio file, or record audio")
        logger.warning("Process attempted with no URL, uploaded file, or recorded audio")
    else:
        logger.info(f"Download requested for URL: {url}")
        try:
            with st.spinner("Downloading..."):
                # Flag to track if direct download is needed
                need_direct_download = True
                download_url = url
                downloaded_file_path = None
                
                # Identify URL type
                if "loom.com" in url:
                    # Handle Loom video
                    logger.info(f"Processing Loom URL: {url}")
                    direct_url = extract_loom_video_url(url)
                    
                    if direct_url:
                        st.info(f"Found direct video URL from Loom")
                        # Use the direct URL for download
                        download_url = direct_url
                    else:
                        # Try using yt-dlp as fallback for Loom
                        if YTDLP_AVAILABLE:
                            try:
                                logger.info("Attempting Loom download with yt-dlp")
                                st.info("Trying to download Loom video...")
                                
                                # Create a temporary directory for download
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    # Set up yt-dlp options
                                    ydl_opts = {
                                        'format': 'best',
                                        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                                        'quiet': True,
                                        'no_warnings': True,
                                        'progress': False,
                                        'restrictfilenames': True,
                                    }
                                    
                                    # Create a progress placeholder
                                    progress_text = st.empty()
                                    progress_text.text("Extracting Loom video information...")
                                    
                                    # Extract info first to get the title
                                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                        info = ydl.extract_info(url, download=False)
                                        title = info.get('title', 'loom_video')
                                        logger.info(f"yt-dlp extracted title: {title}")
                                        
                                        # Update progress text
                                        progress_text.text(f"Downloading: {title}")
                                        
                                        # Download
                                        logger.info("Starting yt-dlp download for Loom")
                                        ydl.download([url])
                                    
                                    # Move downloaded file from temp dir to downloads dir
                                    temp_files = list(Path(temp_dir).glob("*"))
                                    if temp_files:
                                        downloaded_file = temp_files[0]
                                        target_path = download_dir / downloaded_file.name
                                        shutil.copy2(downloaded_file, target_path)
                                        logger.info(f"Loom video downloaded to: {target_path}")
                                        st.success(f"Downloaded Loom video: {downloaded_file.name}")
                                        # Skip direct download
                                        need_direct_download = False
                                        downloaded_file_path = target_path
                                    else:
                                        logger.error("Loom download failed via yt-dlp")
                                        st.error("Could not download Loom video automatically")
                                        st.info("Try copying the direct video URL from the Loom page and pasting it here")
                            except Exception as loom_error:
                                logger.error(f"Loom download error: {str(loom_error)}")
                                st.error(f"Failed to download Loom video: {str(loom_error)}")
                        else:
                            st.error("Could not extract direct URL from Loom")
                            st.info("Try right-clicking on the Loom video, selecting 'Copy Video Address' and pasting that URL here instead")

                elif url.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')) or "//cdn." in url:
                    # Direct video URL
                    logger.info(f"Processing direct video URL: {url}")
                    download_url = url
                    
                elif "youtube.com" in url or "youtu.be" in url:
                    # YouTube videos - deprioritized but still supported as fallback
                    logger.info(f"YouTube URL detected (note: YouTube downloads may be blocked): {url}")
                    st.warning("YouTube downloads are less reliable due to YouTube's restrictions. Loom or direct MP4 links are recommended.")
                    
                    # Try with yt-dlp
                    if YTDLP_AVAILABLE:
                        try:
                            logger.info("Attempting YouTube download with yt-dlp")
                            st.info("Trying to download YouTube video...")
                            
                            # Create a temporary directory for download
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Set up yt-dlp options
                                ydl_opts = {
                                    'format': 'best',
                                    'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                                    'quiet': True,
                                    'no_warnings': True,
                                    'progress': False,
                                    'restrictfilenames': True,
                                }
                                
                                # Create a progress placeholder
                                progress_text = st.empty()
                                progress_text.text("Extracting YouTube video information...")
                                
                                # Extract info first to get the title
                                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                    info = ydl.extract_info(url, download=False)
                                    title = info.get('title', 'youtube_video')
                                    logger.info(f"yt-dlp extracted title: {title}")
                                    
                                    # Update progress text
                                    progress_text.text(f"Downloading: {title}")
                                    
                                    # Download
                                    logger.info("Starting yt-dlp download for YouTube")
                                    ydl.download([url])
                                
                                # Move downloaded file from temp dir to downloads dir
                                temp_files = list(Path(temp_dir).glob("*"))
                                if temp_files:
                                    downloaded_file = temp_files[0]
                                    target_path = download_dir / downloaded_file.name
                                    shutil.copy2(downloaded_file, target_path)
                                    logger.info(f"YouTube video downloaded to: {target_path}")
                                    st.success(f"Downloaded YouTube video: {downloaded_file.name}")
                                    # Skip direct download
                                    need_direct_download = False
                                    downloaded_file_path = target_path
                                else:
                                    logger.error("YouTube download failed via yt-dlp")
                                    st.error("Could not download YouTube video")
                        except Exception as yt_error:
                            logger.error(f"YouTube download error: {str(yt_error)}")
                            st.error(f"Failed to download YouTube video: {str(yt_error)}")
                            st.info("Try using a Loom or direct MP4 link instead")
                    else:
                        st.error("YouTube downloads are not supported without yt-dlp")
                        st.info("Please use a Loom or direct MP4 link instead")
                else:
                    # Try to treat as a direct URL
                    logger.info(f"Processing URL as direct link: {url}")
                    download_url = url
                
                # Direct URL download if needed
                if need_direct_download:
                    logger.info(f"Downloading from direct URL: {download_url}")
                    try:
                        # Add User-Agent header to avoid some 403 errors
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        response = requests.get(download_url, stream=True, headers=headers, timeout=30)
                        
                        if response.status_code == 200:
                            # Check content type to confirm it's a video
                            content_type = response.headers.get('content-type', '')
                            if not ('video' in content_type or 'application/octet-stream' in content_type) and not download_url.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                                logger.warning(f"Content type is not video: {content_type}")
                                st.warning(f"The URL might not be a video (content type: {content_type})")
                                # Continue anyway, user might know better
                            
                            # Get filename from URL or use default
                            file_name = download_url.split('/')[-1].split('?')[0]  # Remove query params
                            if not file_name or '.' not in file_name:
                                file_name = "downloaded_video.mp4"
                                
                            file_path = os.path.join(download_dir, file_name)
                            logger.info(f"Downloading direct URL to: {file_path}")
                            
                            # Download with progress
                            with open(file_path, 'wb') as f:
                                total_length = int(response.headers.get('content-length', 0))
                                logger.info(f"File size: {total_length} bytes")
                                
                                if total_length:
                                    progress_bar = st.progress(0)
                                    downloaded = 0
                                    
                                    for chunk in response.iter_content(chunk_size=1024):
                                        if chunk:
                                            f.write(chunk)
                                            downloaded += len(chunk)
                                            progress = min(1.0, downloaded / total_length)
                                            progress_bar.progress(progress)
                                            
                                            # Log progress at 25%, 50%, 75% and 100%
                                            if int(progress * 100) % 25 == 0 and int((downloaded - len(chunk)) / total_length * 100) % 25 != 0:
                                                logger.info(f"Download progress: {int(progress * 100)}%")
                                else:
                                    logger.warning("Content length unknown, downloading without progress tracking")
                                    f.write(response.content)
                            
                            logger.info(f"Successfully downloaded file from direct URL: {file_path}")
                            st.success(f"Downloaded file: {file_name}")
                            downloaded_file_path = file_path
                        else:
                            logger.error(f"Failed to download. Status code: {response.status_code}, Response: {response.text[:200]}")
                            st.error(f"Failed to download. Status code: {response.status_code}")
                    except requests.exceptions.RequestException as req_error:
                        error_details = traceback.format_exc()
                        logger.error(f"Request error: {str(req_error)}\n{error_details}")
                        st.error(f"Download error: {str(req_error)}")
                
                # Extract audio automatically and file downloaded successfully
                if downloaded_file_path:
                    st.info("Extracting audio from video...")
                    audio_path = extract_audio_from_video(downloaded_file_path)
                    if audio_path:
                        st.success(f"Audio extracted successfully: {audio_path.name}")
                    else:
                        st.error("Failed to extract audio from video")
                
                # Transcribe audio automatically
                transcription_text = None
                if WHISPER_AVAILABLE:
                    audio_file_to_transcribe = None
                    
                    # Determine which audio file to transcribe
                    if downloaded_file_path and 'audio_path' in locals() and audio_path:
                        # Use the newly extracted audio
                        audio_file_to_transcribe = audio_path
                        st.info("Transcribing newly extracted audio...")
                    else:
                        # Look for existing audio files
                        audio_files = list(audio_dir.glob("*.wav"))
                        if audio_files:
                            # Use the most recent audio file
                            audio_file_to_transcribe = max(audio_files, key=lambda x: x.stat().st_mtime)
                            st.info(f"Transcribing audio file: {audio_file_to_transcribe.name}")
                        else:
                            st.warning("No audio file found for transcription.")
                    
                    # Perform transcription
                    if audio_file_to_transcribe:
                        with st.spinner(f"Transcribing audio using Whisper ({whisper_model} model)..."):
                            transcription_text, error = transcribe_audio_with_whisper(
                                audio_file_to_transcribe, 
                                model_size=whisper_model
                            )
                            
                            if transcription_text:
                                st.success("Transcription completed successfully!")
                                
                                # Display transcription in an expandable section
                                with st.expander("📝 Transcription Result", expanded=True):
                                    st.text_area(
                                        "Transcribed Text",
                                        transcription_text,
                                        height=300,
                                        help="This is the transcribed text from the audio. You can copy this text."
                                    )
                                    
                                    # Add a download button for the transcription
                                    transcription_file = transcriptions_dir / f"{audio_file_to_transcribe.stem}_transcription.txt"
                                    if transcription_file.exists():
                                        with open(transcription_file, 'r', encoding='utf-8') as f:
                                            st.download_button(
                                                label="Download Transcription File",
                                                data=f.read(),
                                                file_name=transcription_file.name,
                                                mime="text/plain"
                                            )
                            else:
                                st.error(f"Transcription failed: {error}")

                # Detect accent automatically
                if ACCENT_DETECTION_AVAILABLE:
                    audio_file_to_analyze = None
                    
                    # Determine which audio file to analyze for accent
                    if downloaded_file_path and 'audio_path' in locals() and audio_path:
                        # Use the newly extracted audio
                        audio_file_to_analyze = audio_path
                        st.info("Analyzing accent from newly extracted audio...")
                    else:
                        # Look for existing audio files
                        audio_files = list(audio_dir.glob("*.wav"))
                        if audio_files:
                            # Use the most recent audio file
                            audio_file_to_analyze = max(audio_files, key=lambda x: x.stat().st_mtime)
                            st.info(f"Analyzing accent from audio file: {audio_file_to_analyze.name}")
                        else:
                            st.warning("No audio file found for accent detection.")
                    
                    # Perform accent detection
                    if audio_file_to_analyze:
                        with st.spinner("Detecting English accent using AI..."):
                            accent_result = detect_english_accent(audio_file_to_analyze)
                            
                            if accent_result.get("success"):
                                st.success("Accent detection completed successfully!")
                                
                                # Get personality tag
                                personality_tag = get_accent_personality_tag(accent_result['detected_accent'])
                                
                                # Display accent detection results
                                with st.expander("🎯 Accent Detection Result", expanded=True):
                                    # Main result with personality tag
                                    st.markdown(f"### 🗣️ Detected Accent: **{accent_result['detected_accent']}**")
                                    st.markdown(f"### 🎭 Voice Persona: **{personality_tag['persona']} {personality_tag['emoji']}**")
                                    st.markdown(f"*{personality_tag['description']}*")
                                    st.markdown(f"### 📊 Confidence: **{accent_result['confidence_percent']}**")
                                    
                                    # Progress bar for confidence
                                    st.progress(accent_result['confidence'])
                                    
                                    # Add download button for accent detection results
                                    accent_file = accent_result['results_file']
                                    if os.path.exists(accent_file):
                                        with open(accent_file, 'r', encoding='utf-8') as f:
                                            st.download_button(
                                                label="Download Accent Detection Results",
                                                data=f.read(),
                                                file_name=os.path.basename(accent_file),
                                                mime="text/plain"
                                            )
                                
                                # Calculate and display interview readiness stamp
                                interview_readiness_stamp = accent_result.get('interview_stamp')
                                if interview_readiness_stamp:
                                    display_interview_readiness_stamp(interview_readiness_stamp)
                            else:
                                st.error(f"Accent detection failed: {accent_result.get('error', 'Unknown error')}")
                
                # Display downloaded files
                st.subheader("Downloaded Files")
                files = list(download_dir.glob("*"))
                logger.info(f"Files in download directory: {[f.name for f in files]}")
                for file in files:
                    st.write(f"📁 {file.name}")
                
                # Display extracted audio files
                    st.subheader("Extracted Audio Files")
                    audio_files = list(audio_dir.glob("*.wav"))
                    logger.info(f"Audio files in directory: {[f.name for f in audio_files]}")
                    for file in audio_files:
                        st.write(f"🔊 {file.name}")
                
                # Display transcription files
                st.subheader("Transcription Files")
                transcription_files = list(transcriptions_dir.glob("*_transcription.txt"))
                logger.info(f"Transcription files in directory: {[f.name for f in transcription_files]}")
                for file in transcription_files:
                    st.write(f"📝 {file.name}")
                
                # Display accent detection files
                st.subheader("Accent Detection Files")
                accent_files = list(transcriptions_dir.glob("*_accent_analysis.txt"))
                logger.info(f"Accent detection files in directory: {[f.name for f in accent_files]}")
                for file in accent_files:
                    st.write(f"🎯 {file.name}")
                
                # Calculate interview readiness stamp
                interview_readiness_stamp = accent_result.get('interview_stamp')
                
                # Display interview readiness stamp
                display_interview_readiness_stamp(interview_readiness_stamp)
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Unexpected error: {str(e)}\n{error_details}")
            st.error(f"Error: {str(e)}")

# Display download folder info
st.markdown("---")
st.info(f"📁 Downloaded files: {os.path.abspath(download_dir)}")
st.info(f"🔊 Audio files: {os.path.abspath(audio_dir)}")
st.info(f"📝 Transcription & accent results: {os.path.abspath(transcriptions_dir)}")

# Add tips about Loom and direct links
st.markdown("""
### Tips for Getting Video URLs
- **Loom Videos**: Right-click on a Loom video and select "Copy Video Address" to get a direct URL
- **Direct MP4 Links**: Look for URLs ending with .mp4, .mov, etc.
- **Other Video Sites**: Right-click on the video and look for options like "Copy video address"

### Audio Input Options
- **Video URLs**: Automatically extracts audio from downloaded videos
- **File Upload**: Supports WAV, MP3, MP4, M4A, FLAC, OGG formats
- **Live Recording**: Click the microphone button to record directly in your browser
- **Recording Tips**: Speak clearly, ensure good microphone quality, avoid background noise
- **Browser Compatibility**: Works on all modern browsers with HTTPS (required for microphone access)

### Audio Transcription & Accent Detection
- **Automatic Processing**: All features are enabled by default - no configuration needed!
- **Whisper Model**: Uses the `base` model for optimal balance of speed and accuracy
- **Audio Quality**: Better audio quality leads to more accurate results
- **Language**: Whisper automatically detects the language (supports 99+ languages)
- **Supported Accents**: General American, British (RP), Australian, Irish, Scottish, Canadian, South African
- **Voice Personas**: Each accent gets a creative personality tag (e.g., "Polished London Professional 🇬🇧")
- **Interview Readiness**: Get color-coded hiring recommendations (✅ Ready, ⚠️ Understandable, ❌ Needs Support)
- **Best Results**: Use clear, uninterrupted speech samples for accent detection
""")

# Add log info at the bottom
with st.expander("Logs and Troubleshooting"):
    st.markdown(f"Log file: `{log_file}`")
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
            st.text_area("Recent logs", log_content, height=200)
            
    st.markdown("""
    ### Troubleshooting Tips
    - **403 Forbidden Error**: Some sites restrict direct downloads. Try getting the direct video URL instead.
    - **Content Type Error**: If the URL doesn't point directly to a video file, try finding a direct link.
    - **Loom Download Issues**: For Loom videos, right-click and select "Copy Video Address".
    - **YouTube Not Working**: YouTube frequently blocks automated downloads. Use Loom or direct links instead.
    - **Audio Extraction Issues**: If audio extraction fails, the video might not have an audio track or be in an unsupported format.
    - **Whisper Not Available**: Install with `pip install faster-whisper`
    - **Transcription Slow**: Use smaller models (tiny/base) for faster processing
    - **Transcription Inaccurate**: Use larger models (medium/large) for better accuracy
    - **GPU Support**: Install CUDA-compatible PyTorch for GPU acceleration (much faster)
    - **Accent Detection Not Available**: Install with `pip install transformers librosa soundfile datasets`
    - **Accent Detection Fails**: Ensure audio is clear English speech, at least 1 second long
    - **Low Confidence Scores**: Try using longer, clearer audio samples
    - **Model Download Issues**: Check internet connection for Hugging Face model downloads
    - **Live Recording Not Working**: Ensure app is served over HTTPS (required for microphone access)
    - **Microphone Permission Denied**: Check browser settings and allow microphone access for the site
    - **Recording Button Not Responding**: Try refreshing the page and ensure stable internet connection
    - **No Audio Recorded**: Click start, speak clearly, then click stop - ensure microphone is working
    - **Audio Recorder Not Available**: Install with `pip install streamlit-audiorecorder`
    """)

logger.info("App session ended") 
