# 🎥 Video Processor & AI Analyzer

**Download videos → Extract audio → Transcribe speech → Detect accents → Get voice personas** - All automatically!

A comprehensive Streamlit application that processes video and audio content with AI-powered transcription, accent detection, personality mapping, and interview readiness assessment. Perfect for HR professionals, language coaches, and content creators.

## ✨ Key Features

### 🚀 **Fully Automated Processing**
- **Zero Configuration Required**: All features enabled by default
- **One-Click Processing**: No checkboxes or settings to configure
- **Smart Defaults**: Optimized settings for best performance
- **Instant Results**: Automatic transcription and accent analysis

### 🎤 **Multiple Input Methods**
1. **Video URLs**: Loom, direct MP4 links, and more
2. **File Upload**: WAV, MP3, MP4, M4A, FLAC, OGG formats  
3. **Live Recording**: Browser-based microphone recording (HTTPS required)

### 🎯 **AI-Powered Accent Detection**
- **7 English Accents**: General American, British (RP), Australian, Irish, Scottish, Canadian, South African
- **Confidence Scoring**: Percentage confidence for each prediction
- **Smart Analysis**: Combines AI models with audio feature analysis
- **Quality Validation**: Automatic audio quality checks

### 🎭 **Voice Personality Mapping**
Transform accent detection into engaging personality profiles:
- **British (RP)** → "Polished London Professional 🇬🇧"
- **General American** → "Confident American Go-Getter 🇺🇸"  
- **Australian** → "Friendly Aussie Vibes 🇦🇺"
- **Irish** → "Charming Irish Storyteller 🇮🇪"
- **Scottish** → "Bold Scottish Character 🏴󠁧󠁢󠁳󠁣󠁴󠁿"
- **Canadian** → "Polite Canadian Diplomat 🇨🇦"
- **South African** → "Dynamic South African Leader 🇿🇦"

### 🧑‍🏫 **Interview Readiness Assessment**
Color-coded hiring recommendations based on communication clarity:

- **✅ Ready (80+ points)**: "Ready for client-facing roles"
- **⚠️ Understandable (60-79 points)**: "Understandable but may benefit from coaching"  
- **❌ Needs Support (<60 points)**: "Needs further English communication practice"

*Scoring considers accent clarity, confidence, audio quality, and fluency*

### 📝 **Advanced Transcription**
- **Faster-Whisper Integration**: Optimized for speed and compatibility
- **99+ Languages**: Automatic language detection
- **Detailed Output**: Timestamps, segments, and confidence scores
- **Multiple Formats**: View in-app or download as text files

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd accent-detector
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Access the app:** Open `http://localhost:8501` in your browser

### Usage

#### Option 1: Video URL Processing
1. Paste a video URL (Loom, direct MP4, etc.)
2. Click "🚀 Process"
3. Get automatic transcription, accent detection, and personality analysis

#### Option 2: Audio File Upload  
1. Click "Choose an audio file"
2. Upload your audio file
3. Click "🎵 Process Audio"
4. Receive instant AI analysis

#### Option 3: Live Recording
1. Click "🎤 Start Recording" 
2. Speak clearly into your microphone
3. Click "⏹️ Stop Recording"
4. Get immediate transcription and accent analysis

*Note: Live recording requires HTTPS (works automatically on Streamlit Cloud)*

## 📊 What You Get

### Comprehensive Analysis Package
- **📝 Full Transcription**: Complete text with timestamps
- **🎯 Accent Detection**: Primary accent with confidence score
- **🎭 Voice Persona**: Creative personality tag and description
- **🧑‍🏫 Interview Assessment**: Color-coded hiring recommendation with score
- **📁 Downloadable Results**: All analysis saved to text files

### Sample Output
```
=== ACCENT DETECTION RESULTS ===
Detected Accent: British (RP)
Confidence: 87.3%

=== VOICE PERSONA ===
Personality Tag: Polished London Professional 🇬🇧
Description: Sophisticated, articulate, and commanding presence

=== INTERVIEW READINESS ASSESSMENT ===
Readiness Level: ✅ Ready for client-facing roles
Score: 89.2/100
Recommendation: Excellent for customer service, sales, management, and public-facing positions
```

## 🛠️ Technical Specifications

### AI Models Used
- **Transcription**: Faster-Whisper (base model) - optimal speed/accuracy balance
- **Accent Detection**: Wav2Vec2-based classification with feature fallback
- **Audio Processing**: 16kHz mono conversion for optimal analysis

### Supported Formats
- **Video**: MP4, MOV, AVI, MKV, WebM, Loom URLs
- **Audio**: WAV, MP3, MP4, M4A, FLAC, OGG
- **Output**: Text files with detailed analysis

### Performance
- **Processing Time**: ~2-5 minutes for 10-minute videos
- **Audio Quality**: Optimized for clear speech recognition
- **Browser Support**: All modern browsers (Chrome, Firefox, Safari, Edge)

## 📁 File Organization

```
accent-detector/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies  
├── packages.txt             # System dependencies (for deployment)
├── downloads/               # Downloaded video files
├── audio/                   # Extracted/uploaded audio files
├── transcriptions/          # Analysis results and transcriptions
└── logs/                    # Application logs
```

## 🌐 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy automatically with HTTPS
4. Live recording works out-of-the-box

### Local Development
- Live recording requires HTTPS for microphone access
- Use tools like `ssl-proxy` for local HTTPS testing
- All other features work on HTTP

## 🔧 Troubleshooting

### Live Recording Issues
- **Not Working**: Ensure app is served over HTTPS
- **Permission Denied**: Check browser microphone permissions
- **Button Unresponsive**: Refresh page, check internet connection

### Audio Processing Issues  
- **Poor Transcription**: Use clearer audio, reduce background noise
- **Low Confidence**: Try longer speech samples (3+ seconds)
- **Model Download Fails**: Check internet connection

### Video Download Issues
- **Loom Videos**: Right-click → "Copy Video Address"
- **403 Errors**: Try direct video URLs instead
- **YouTube Blocked**: Use Loom or direct MP4 links

## 📦 Dependencies

### Core Libraries
```
streamlit>=1.32.0           # Web interface
faster-whisper>=1.0.0       # Speech recognition
transformers>=4.21.0        # AI models
torch>=2.0.0               # ML framework
librosa>=0.10.0            # Audio processing
streamlit-audiorecorder     # Live recording (cloud-compatible)
```

### Audio/Video Processing
```
moviepy>=1.0.3             # Video processing
soundfile>=0.12.0          # Audio I/O
ffmpeg-python>=0.2.0       # Media processing
yt-dlp>=2023.1.6          # Video downloads
```

## 🎯 Use Cases

### For HR Professionals
- **Candidate Assessment**: Evaluate communication skills objectively
- **Interview Preparation**: Help candidates understand their speaking style
- **Diversity & Inclusion**: Appreciate accent diversity while assessing clarity

### For Language Coaches
- **Accent Training**: Identify specific accent patterns for targeted coaching
- **Progress Tracking**: Monitor improvement over time
- **Personalized Feedback**: Provide specific recommendations based on analysis

### For Content Creators
- **Voice Analysis**: Understand your speaking style and persona
- **Audience Targeting**: Align voice persona with content strategy
- **Quality Control**: Ensure clear communication in recordings

## 📄 License

This project is for educational and professional use. Please respect copyright and terms of service of video platforms.

---

**🚀 Ready to analyze your voice? Deploy on Streamlit Cloud or run locally to get started!** 