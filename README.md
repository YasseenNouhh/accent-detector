# Video Downloader & Audio Transcription App

A Streamlit-based application that downloads videos from various sources and transcribes audio using OpenAI's Whisper model.

## Features

### 🎥 Video Download
- Download videos from Loom URLs
- Support for direct MP4/video links
- YouTube video support (limited due to restrictions)
- Fallback support using yt-dlp

### 🔊 Audio Extraction
- Extract audio from downloaded videos
- Convert to WAV format for optimal transcription
- Automatic audio processing

### 📝 Audio Transcription
- **OpenAI Whisper Integration**: State-of-the-art speech recognition
- **Multiple Model Sizes**: Choose between speed and accuracy
- **Multi-language Support**: Automatically detects 99+ languages
- **Multiple Input Methods**: 
  - Transcribe audio from downloaded videos
  - Upload audio files directly
  - Process existing audio files
- **Rich Output**: 
  - View transcription in the web interface
  - Download as text file with timestamps
  - Detailed segment-by-segment breakdown

### 🎯 Accent Detection
- **AI-Powered Analysis**: Uses Hugging Face's `wav2vec2-large-xlsr-53-english-accent-classifier`
- **Multiple English Accents**: Detects American, British, Australian, Canadian, Indian, and more
- **Confidence Scoring**: Provides percentage confidence for predictions
- **Smart Preprocessing**: Automatically converts audio to optimal format (16kHz mono)
- **Quality Checks**: Validates audio duration and volume levels
- **Detailed Results**: 
  - Primary accent prediction with confidence score
  - Top 5 accent predictions
  - Technical analysis details
  - Downloadable results file

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. For GPU Acceleration (Optional but Recommended)

If you have a CUDA-compatible GPU, install PyTorch with CUDA support for much faster transcription:

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Run the Application

```bash
streamlit run app.py
```

## Usage

### Video Download & Transcription
1. Enter a video URL (Loom, direct MP4, etc.)
2. Check "Extract audio to WAV format" (recommended for transcription)
3. Check "Transcribe audio to text using Whisper"
4. Select your preferred Whisper model size
5. Click "Download"

### Direct Audio Transcription
1. Click "Choose an audio file" to upload an audio file
2. Check "Transcribe audio to text using Whisper"
3. Select your preferred Whisper model size
4. Click "Process Audio"

### Accent Detection
1. **From Video**: Download video → Extract audio → Check "Detect English accent using AI"
2. **From Upload**: Upload audio file → Check "Detect English accent using AI"
3. **From Existing**: Use existing audio files in your `audio/` folder
4. Click "Download" or "Process Audio"
5. View results with confidence scores and download detailed analysis

## Whisper Model Comparison

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~39 MB | Fastest | Basic | Quick transcription, low-quality audio |
| `base` | ~74 MB | Fast | Good | **Recommended balance** |
| `small` | ~244 MB | Medium | Better | Higher accuracy needs |
| `medium` | ~769 MB | Slow | High | Professional transcription |
| `large` | ~1550 MB | Very Slow | Best | Maximum accuracy required |

## File Structure

```
accent-detector/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── downloads/            # Downloaded video files
├── audio/               # Extracted audio files
├── transcriptions/      # Transcription & accent detection results
└── logs/               # Application logs
```

## Supported Formats

### Video Input
- Loom videos
- Direct MP4, MOV, AVI, MKV, WebM links
- YouTube videos (limited)

### Audio Input
- WAV, MP3, MP4, M4A, FLAC, OGG

## Troubleshooting

### Common Issues

1. **Whisper Not Available**
   ```bash
   pip install openai-whisper torch torchaudio
   ```

2. **Slow Transcription**
   - Use smaller models (tiny/base)
   - Install GPU-accelerated PyTorch
   - Ensure sufficient RAM

3. **Download Issues**
   - For Loom: Right-click video → "Copy Video Address"
   - Use direct video URLs when possible
   - Check internet connection

4. **Audio Extraction Fails**
   - Video might not have audio track
   - Try different video source
   - Check video file integrity

### Performance Tips

1. **GPU Acceleration**: Install CUDA-compatible PyTorch for 5-10x faster transcription
2. **Model Selection**: Use `base` model for best speed/accuracy balance
3. **Audio Quality**: Higher quality audio = better transcription accuracy
4. **File Size**: Larger audio files take longer to process

## Dependencies

- `streamlit`: Web interface
- `openai-whisper`: Speech recognition
- `faster-whisper`: Alternative speech recognition (Windows-friendly)
- `torch` & `torchaudio`: ML framework
- `transformers`: Hugging Face models for accent detection
- `librosa`: Audio processing and analysis
- `soundfile`: Audio file I/O
- `datasets`: Hugging Face datasets support
- `moviepy`: Video/audio processing
- `requests`: HTTP downloads
- `yt-dlp`: Video download fallback

## License

This project is for educational and personal use. Respect copyright and terms of service of video platforms.
