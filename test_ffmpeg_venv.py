#!/usr/bin/env python3
"""
Test script to verify FFmpeg works in virtual environment
"""

import os
import subprocess
import sys
from pathlib import Path

def test_ffmpeg_access():
    """Test if FFmpeg is accessible from Python"""
    print("🔍 Testing FFmpeg access from Python...")
    
    try:
        # Test ffmpeg
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg accessible from Python")
            ffmpeg_version = result.stdout.split('\n')[0]
            print(f"   Version: {ffmpeg_version}")
        else:
            print("❌ FFmpeg not accessible from Python")
            return False
    except Exception as e:
        print(f"❌ FFmpeg test failed: {e}")
        return False
    
    try:
        # Test ffprobe
        result = subprocess.run(['ffprobe', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFprobe accessible from Python")
        else:
            print("❌ FFprobe not accessible from Python")
            return False
    except Exception as e:
        print(f"❌ FFprobe test failed: {e}")
        return False
    
    return True

def test_moviepy_import():
    """Test MoviePy import and FFmpeg configuration"""
    print("\n🔍 Testing MoviePy import...")
    
    # Configure FFmpeg path first
    try:
        result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip().split('\n')[0]
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            
            # Set environment variables for MoviePy
            os.environ['FFMPEG_BINARY'] = ffmpeg_path
            os.environ['FFPROBE_BINARY'] = os.path.join(ffmpeg_dir, 'ffprobe.exe')
            
            print(f"✅ FFmpeg path configured: {ffmpeg_path}")
    except Exception as e:
        print(f"⚠️ Could not configure FFmpeg path: {e}")
    
    # Test MoviePy import
    try:
        from moviepy import VideoFileClip
        print("✅ MoviePy 2.x imported successfully")
        return True
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip
            print("✅ MoviePy 1.x imported successfully")
            return True
        except ImportError as e:
            print(f"❌ MoviePy import failed: {e}")
            return False
    except Exception as e:
        print(f"❌ MoviePy import error: {e}")
        return False

def test_audio_libraries():
    """Test audio processing libraries"""
    print("\n🔍 Testing audio processing libraries...")
    
    libraries = [
        ('librosa', 'librosa'),
        ('soundfile', 'soundfile'),
        ('faster-whisper', 'faster_whisper'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('streamlit-audiorecorder', 'audiorecorder')
    ]
    
    all_good = True
    for lib_name, import_name in libraries:
        try:
            __import__(import_name)
            print(f"✅ {lib_name}")
        except ImportError:
            print(f"❌ {lib_name} - Not installed")
            all_good = False
        except Exception as e:
            print(f"⚠️ {lib_name} - Import error: {e}")
            all_good = False
    
    return all_good

def test_environment():
    """Test Python environment"""
    print("\n🔍 Testing Python environment...")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️ Not running in virtual environment")
    
    return True

def main():
    """Run all tests"""
    print("🧪 FFmpeg Virtual Environment Test")
    print("=" * 40)
    
    tests = [
        ("Environment", test_environment),
        ("FFmpeg Access", test_ffmpeg_access),
        ("MoviePy Import", test_moviepy_import),
        ("Audio Libraries", test_audio_libraries)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your environment is ready.")
        print("You can now run: streamlit run app.py")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        print("Try running: pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...") 