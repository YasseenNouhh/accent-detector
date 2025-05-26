# Streamlit Cloud Deployment Troubleshooting Guide

## Common Issues and Solutions

### 1. Package Installation Errors

#### Issue: Comments in packages.txt causing installation failures
```
E: Unable to locate package #
E: Unable to locate package FFmpeg
E: Unable to locate package for
```

**Solution**: Remove all comments from `packages.txt`
```bash
# ❌ Wrong - causes errors
# FFmpeg for video processing
ffmpeg

# ✅ Correct - no comments
ffmpeg
libsndfile1
libasound2-dev
portaudio19-dev
libgomp1
```

#### Issue: Package version not found
```
ERROR: Could not find a version that satisfies the requirement streamlit-audiorecorder>=0.4.0
```

**Solution**: Use exact versions that exist on PyPI
```bash
# ❌ Wrong - version doesn't exist
streamlit-audiorecorder>=0.4.0

# ✅ Correct - use available version
streamlit-audiorecorder==0.0.6
```

### 2. Audio Recording Issues

#### Issue: Audio recorder not working
If you encounter issues with `streamlit-audiorecorder`, try the alternative package:

**Option 1**: Use `streamlit-audiorecorder==0.0.6` (current setup)
```python
from audiorecorder import audiorecorder
audio = audiorecorder("Click to record", "Click to stop recording")
```

**Option 2**: Use `audio-recorder-streamlit==0.0.10` (newer alternative)
```python
from audio_recorder_streamlit import audio_recorder
audio_bytes = audio_recorder()
```

To switch to the alternative:
1. Replace `requirements.txt` with `requirements_alternative.txt`
2. Update your code to use the new import

### 3. Model Loading Issues

#### Issue: Accent detection models fail to load
```
❌ ylacombe/accent-classifier failed: HTTP 404
❌ dima806/multiple_accent_classification failed: Network error
```

**Solution**: The system automatically falls back to rule-based detection
- This is expected behavior
- The app will still work with the fallback system
- Models will be cached after first successful download

#### Issue: Transformers version conflicts
```
ERROR: transformers 4.21.0 has requirement torch>=1.9.0, but you have torch 1.8.0
```

**Solution**: Update transformers version
```bash
# In requirements.txt
transformers>=4.35.0  # Updated version
torch>=2.0.0          # Compatible torch version
```

### 4. Memory and Performance Issues

#### Issue: App runs out of memory
```
❌ CUDA out of memory
```

**Solution**: The system automatically handles this
- Falls back to CPU processing
- Reduces model complexity if needed
- Uses smaller audio chunks

#### Issue: Slow model loading
**Solution**: Models are cached after first download
- First run may be slower
- Subsequent runs will be faster
- Consider using lighter models for production

### 5. Audio Processing Issues

#### Issue: FFmpeg not found
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**Solution**: Ensure FFmpeg is in packages.txt
```bash
# packages.txt
ffmpeg
libsndfile1
libasound2-dev
portaudio19-dev
libgomp1
```

#### Issue: Audio format not supported
**Solution**: The app handles multiple formats
- WAV, MP3, M4A, FLAC are supported
- Audio is automatically converted to 16kHz mono
- Maximum 30 seconds for processing

### 6. Network and Security Issues

#### Issue: HTTPS required for audio recording
```
Error: navigator.mediaDevices is undefined
```

**Solution**: Streamlit Cloud automatically provides HTTPS
- No action needed for Streamlit Cloud
- For local development, use `localhost` not IP addresses

#### Issue: Models can't download due to firewall
**Solution**: The app has multiple fallback layers
- Tries multiple model sources
- Falls back to rule-based system
- Always provides a result

## Deployment Checklist

### Before Deploying
- [ ] Remove all comments from `packages.txt`
- [ ] Use exact package versions in `requirements.txt`
- [ ] Test locally with `streamlit run app.py`
- [ ] Verify audio recording works locally

### After Deploying
- [ ] Check deployment logs for errors
- [ ] Test audio recording functionality
- [ ] Verify accent detection works
- [ ] Check model loading (may take time on first run)

### If Issues Persist
1. **Check the logs**: Look for specific error messages
2. **Try the alternative audio recorder**: Use `requirements_alternative.txt`
3. **Restart the app**: Sometimes helps with model loading
4. **Clear cache**: In Streamlit Cloud settings

## File Structure for Deployment

```
accent-detector/
├── app.py                          # Main application
├── packages.txt                    # System packages (no comments!)
├── requirements.txt                # Python packages (exact versions)
├── requirements_alternative.txt    # Alternative audio recorder
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── audio/                         # Audio files directory
├── transcriptions/                # Transcription outputs
└── README.md                      # Documentation
```

## Testing Commands

### Local Testing
```bash
# Test the app locally
streamlit run app.py

# Test accent models
python test_accent_models.py

# Check package versions
pip list | grep -E "(streamlit|torch|transformers)"
```

### Deployment Testing
```bash
# Test with exact requirements
pip install -r requirements.txt

# Test alternative audio recorder
pip install -r requirements_alternative.txt
```

## Support

If you continue to have issues:

1. **Check the error logs** in Streamlit Cloud
2. **Try the alternative requirements** file
3. **Restart the app** in Streamlit Cloud
4. **Check model availability** - some models may be temporarily unavailable

The app is designed to be robust with multiple fallback mechanisms, so it should always provide some level of functionality even if individual components fail. 