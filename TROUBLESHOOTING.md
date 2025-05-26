# Troubleshooting Guide - Accent Detector

## Quick Fix (Recommended)

Run the automated fix script to resolve most common issues:

```bash
python fix_issues.py
```

This script will:
- ✅ Check all required packages
- ✅ Install missing dependencies
- ✅ Set up FFmpeg automatically
- ✅ Create required directories
- ✅ Clean up invalid files
- ✅ Test basic functionality

## Common Issues & Solutions

### 1. FFmpeg/FFprobe Not Found ⚠️

**Error:** `FileNotFoundError: [WinError 2] The system cannot find the file specified`

**Solution:**
```bash
python setup_ffmpeg.py
```

**Manual Installation (Windows):**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH
4. Restart command prompt

### 2. Audio Recording Fails 🎤

**Error:** Audio recorder crashes or doesn't work

**Cause:** Missing FFmpeg/FFprobe

**Solution:**
```bash
python setup_ffmpeg.py
```

### 3. YouTube Downloads Blocked 🚫

**Error:** `HTTP Error 403: Forbidden`

**Solution:** YouTube blocks automated downloads. Use alternatives:
- ✅ Upload audio files directly
- ✅ Use Loom videos
- ✅ Use live recording feature
- ✅ Find direct MP4 links

### 4. Video Processing Errors 📹

**Error:** `MoviePy error: failed to read the duration`

**Causes:**
- Downloaded HTML instead of video (common with YouTube)
- Invalid or corrupted video file
- Missing FFmpeg

**Solutions:**
1. Install FFmpeg: `python setup_ffmpeg.py`
2. Use direct video links instead of YouTube
3. Upload audio files directly

### 5. Missing Python Packages 📦

**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```bash
pip install -r requirements.txt
```

Or run the fix script:
```bash
python fix_issues.py
```

### 6. Wav2Vec2 Model Issues 🤖

**Error:** Model download fails or accent detection unavailable

**Solutions:**
1. Check internet connection
2. Wait and retry (Hugging Face rate limiting)
3. App will use fallback mode automatically

### 7. Transcription Issues 📝

**Error:** Whisper models fail to load

**Solutions:**
1. Check internet connection
2. Install faster-whisper: `pip install faster-whisper`
3. App will use fallback mode if models unavailable

## Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up FFmpeg
```bash
python setup_ffmpeg.py
```

### 3. Run Quick Fix
```bash
python fix_issues.py
```

### 4. Start the App
```bash
streamlit run app.py
```

## Alternative Input Methods

If video downloads fail, try these alternatives:

### 1. Upload Audio Files 📁
- Supports: WAV, MP3, MP4, M4A, FLAC, OGG
- No internet required
- Most reliable method

### 2. Live Recording 🎤
- Record directly in browser
- Requires HTTPS (microphone access)
- Requires FFmpeg for processing

### 3. Loom Videos 📹
- Right-click video → "Copy Video Address"
- Paste the direct URL
- More reliable than YouTube

## Environment Check

Run this to check your setup:

```python
# Test script
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except:
    print("❌ PyTorch not installed")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except:
    print("❌ Transformers not installed")

try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
    print("✅ FFmpeg installed" if result.returncode == 0 else "❌ FFmpeg not found")
except:
    print("❌ FFmpeg not found")
```

## Getting Help

1. **Run the fix script first:** `python fix_issues.py`
2. **Check the troubleshooting section** in the app
3. **Look at the logs** in the `logs/` directory
4. **Try alternative input methods** (upload files, live recording)

## Success Indicators

When everything is working correctly, you should see:

```
✅ PyTorch version: 2.7.0+cpu
✅ Accent detection libraries loaded successfully
✅ Using faster-whisper for transcription
✅ Audio recorder available
```

## Performance Tips

- **Use shorter audio clips** (< 30 seconds) for faster processing
- **Ensure good audio quality** for better accent detection
- **Use direct file uploads** when possible (most reliable)
- **Restart the app** if you encounter temporary issues 