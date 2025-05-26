# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### 1. Prerequisites
- GitHub repository with your code
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### 2. Required Files ✅
Your repository should contain these files (all included):

```
accent-detector/
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (FFmpeg, etc.)
├── .streamlit/config.toml          # Streamlit configuration
├── runtime.txt                     # Python version (optional)
└── README.md                       # Documentation
```

### 3. Deploy Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Wait for Build** (5-10 minutes)
   - Streamlit Cloud will install all dependencies
   - FFmpeg will be installed via `packages.txt`
   - All Python packages will be installed via `requirements.txt`

### 4. Cloud Optimizations ⚡

The app automatically detects Streamlit Cloud and optimizes:

- **Memory Usage**: Uses int8 quantization for Whisper models
- **FFmpeg Detection**: Automatically finds FFmpeg on Linux
- **Worker Limits**: Reduces concurrent workers for stability
- **File Size Limits**: Optimized for 200MB upload limit

### 5. Features Available on Cloud

✅ **Fully Supported:**
- Audio file uploads (WAV, MP3, MP4, M4A, FLAC, OGG)
- Live audio recording (requires HTTPS - automatic on Streamlit Cloud)
- Whisper transcription (faster-whisper)
- Wav2Vec2 accent detection
- Voice persona generation
- Interview readiness assessment
- Direct MP4/video URL downloads

⚠️ **Limited:**
- YouTube downloads (blocked by platform restrictions)
- Very large files (>200MB limit)

### 6. Environment Variables (Optional)

You can set these in Streamlit Cloud settings:

```
STREAMLIT_CLOUD=true          # Auto-detected
WHISPER_MODEL_SIZE=base       # Default model size
MAX_AUDIO_LENGTH=300          # Max audio length in seconds
```

### 7. Troubleshooting

**Common Issues:**

1. **Build Fails**
   - Check `requirements.txt` for version conflicts
   - Ensure `packages.txt` has correct system dependencies

2. **FFmpeg Not Found**
   - Verify `ffmpeg` is in `packages.txt`
   - Check app logs for FFmpeg detection messages

3. **Memory Issues**
   - App automatically uses smaller models on cloud
   - Consider using "tiny" Whisper model for very limited memory

4. **Upload Issues**
   - Files must be <200MB (Streamlit Cloud limit)
   - Use direct audio files instead of large videos

### 8. Performance Tips

- **Audio Files**: Upload directly for fastest processing
- **Video Files**: Use shorter clips (<5 minutes) for better performance
- **Live Recording**: Works best with good microphone quality
- **Model Selection**: App automatically optimizes model size for cloud

### 9. Monitoring

Check your app health:
- Streamlit Cloud dashboard shows resource usage
- App logs available in cloud console
- Built-in error handling with fallback modes

### 10. Updates

To update your deployed app:
```bash
git add .
git commit -m "Update app"
git push origin main
```

Streamlit Cloud will automatically redeploy within minutes.

---

## Ready to Deploy! 🚀

Your app is fully configured for Streamlit Cloud deployment. Just push to GitHub and deploy!

**Live Demo**: Once deployed, your app will be available at:
`https://your-app-name.streamlit.app` 