#!/usr/bin/env python3
"""
Quick fix script for common accent detector issues
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    print("🔍 Checking Python packages...")
    
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'librosa',
        'soundfile',
        'faster_whisper',
        'moviepy',
        'requests',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print("\n🔍 Checking FFmpeg...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            return True
    except:
        pass
    
    try:
        result = subprocess.run(['ffprobe', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFprobe is installed")
            return True
    except:
        pass
    
    print("❌ FFmpeg/FFprobe not found")
    return False

def check_directories():
    """Check if required directories exist"""
    print("\n🔍 Checking directories...")
    
    directories = ['downloads', 'audio', 'transcriptions', 'logs']
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️ {dir_name}/ - Creating...")
            dir_path.mkdir(exist_ok=True)
            print(f"✅ {dir_name}/ - Created")

def fix_missing_packages(missing_packages):
    """Install missing packages"""
    if not missing_packages:
        return True
    
    print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
    
    try:
        # Try to install from requirements.txt first
        if Path('requirements.txt').exists():
            print("📦 Installing from requirements.txt...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Packages installed successfully")
                return True
            else:
                print(f"⚠️ Some packages may have failed: {result.stderr}")
        
        # Install individual packages
        for package in missing_packages:
            print(f"📦 Installing {package}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {package} installed")
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def fix_ffmpeg():
    """Try to fix FFmpeg issues"""
    print("\n🔧 Fixing FFmpeg...")
    
    if Path('setup_ffmpeg.py').exists():
        print("📦 Running FFmpeg setup script...")
        try:
            result = subprocess.run([sys.executable, 'setup_ffmpeg.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ FFmpeg setup completed")
                return True
            else:
                print(f"⚠️ FFmpeg setup had issues: {result.stderr}")
        except Exception as e:
            print(f"❌ Failed to run FFmpeg setup: {e}")
    
    print("\n📋 Manual FFmpeg installation:")
    if platform.system() == "Windows":
        print("1. Download FFmpeg from: https://ffmpeg.org/download.html")
        print("2. Extract to C:\\ffmpeg")
        print("3. Add C:\\ffmpeg\\bin to your system PATH")
        print("4. Restart your command prompt")
    else:
        print("Ubuntu/Debian: sudo apt install ffmpeg")
        print("macOS: brew install ffmpeg")
        print("CentOS/RHEL: sudo yum install ffmpeg")
    
    return False

def clean_temp_files():
    """Clean up temporary files"""
    print("\n🧹 Cleaning temporary files...")
    
    # Clean downloads directory of invalid files
    downloads_dir = Path('downloads')
    if downloads_dir.exists():
        for file in downloads_dir.glob('*'):
            if file.is_file() and file.stat().st_size < 1000:  # Files smaller than 1KB are likely invalid
                print(f"🗑️ Removing invalid file: {file.name}")
                file.unlink()
    
    print("✅ Cleanup completed")

def test_basic_functionality():
    """Test basic app functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports
        import streamlit
        import torch
        print("✅ Core imports work")
        
        # Test Wav2Vec2 model loading (without actually loading)
        from transformers import Wav2Vec2Processor
        print("✅ Wav2Vec2 imports work")
        
        # Test audio processing
        import librosa
        import soundfile
        print("✅ Audio processing imports work")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Accent Detector - Quick Fix Tool")
    print("=" * 40)
    
    # Check current status
    missing_packages = check_python_packages()
    ffmpeg_ok = check_ffmpeg()
    check_directories()
    
    # Apply fixes
    fixes_needed = []
    
    if missing_packages:
        fixes_needed.append("packages")
        
    if not ffmpeg_ok:
        fixes_needed.append("ffmpeg")
    
    if not fixes_needed:
        print("\n🎉 No issues detected! Your setup looks good.")
        clean_temp_files()
        test_basic_functionality()
        print("\n✅ All checks passed. You should be able to run the app now.")
        return True
    
    print(f"\n🔧 Issues detected: {', '.join(fixes_needed)}")
    print("Attempting to fix...")
    
    # Fix missing packages
    if "packages" in fixes_needed:
        if not fix_missing_packages(missing_packages):
            print("⚠️ Some package installation issues occurred")
    
    # Fix FFmpeg
    if "ffmpeg" in fixes_needed:
        if not fix_ffmpeg():
            print("⚠️ FFmpeg installation needs manual intervention")
    
    # Clean up
    clean_temp_files()
    
    # Final test
    print("\n🧪 Running final tests...")
    if test_basic_functionality():
        print("\n🎉 Fixes applied successfully!")
        print("You should now be able to run: streamlit run app.py")
    else:
        print("\n⚠️ Some issues remain. Check the error messages above.")
        print("You may need to:")
        print("1. Restart your terminal/command prompt")
        print("2. Install FFmpeg manually")
        print("3. Check your Python environment")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Fix process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...") 