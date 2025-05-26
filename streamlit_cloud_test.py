#!/usr/bin/env python3
"""
Streamlit Cloud Deployment Test
This script tests if all dependencies are working correctly on Streamlit Cloud
"""

import sys
import platform
import subprocess

def test_system_info():
    """Test basic system information"""
    print("=== SYSTEM INFO ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print()

def test_ffmpeg():
    """Test FFmpeg installation"""
    print("=== FFMPEG TEST ===")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and working")
            print(f"Version: {result.stdout.split()[2]}")
        else:
            print("❌ FFmpeg command failed")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
    except Exception as e:
        print(f"❌ FFmpeg test error: {e}")
    print()

def test_python_packages():
    """Test Python package imports"""
    print("=== PYTHON PACKAGES TEST ===")
    
    packages_to_test = [
        ("streamlit", "Streamlit web framework"),
        ("requests", "HTTP requests library"),
        ("moviepy", "Video processing"),
        ("torch", "PyTorch ML framework"),
        ("transformers", "Hugging Face transformers"),
        ("faster_whisper", "Faster Whisper ASR"),
        ("librosa", "Audio analysis"),
        ("soundfile", "Audio file I/O"),
        ("numpy", "Numerical computing"),
        ("scipy", "Scientific computing"),
        ("sklearn", "Machine learning"),
    ]
    
    for package, description in packages_to_test:
        try:
            __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError as e:
            print(f"❌ {package}: {description} - {e}")
        except Exception as e:
            print(f"⚠️ {package}: {description} - {e}")
    print()

def test_optional_packages():
    """Test optional packages"""
    print("=== OPTIONAL PACKAGES TEST ===")
    
    optional_packages = [
        ("yt_dlp", "YouTube downloader"),
        ("pydub", "Audio manipulation"),
        ("audiorecorder", "Streamlit audio recorder"),
        ("datasets", "Hugging Face datasets"),
    ]
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"⚠️ {package}: {description} - Optional, not critical")
        except Exception as e:
            print(f"⚠️ {package}: {description} - {e}")
    print()

def test_model_access():
    """Test if we can access Hugging Face models"""
    print("=== MODEL ACCESS TEST ===")
    try:
        from transformers import Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        print("✅ Hugging Face model access working")
    except Exception as e:
        print(f"⚠️ Model access issue: {e}")
    print()

if __name__ == "__main__":
    print("🧪 STREAMLIT CLOUD DEPLOYMENT TEST")
    print("=" * 50)
    
    test_system_info()
    test_ffmpeg()
    test_python_packages()
    test_optional_packages()
    test_model_access()
    
    print("🎯 TEST COMPLETE")
    print("If you see mostly ✅ marks, your deployment should work!")
    print("If you see ❌ marks, check your packages.txt and requirements.txt files.") 