#!/usr/bin/env python3
"""
Setup script to install FFmpeg on Windows for the accent detector app
"""

import os
import sys
import subprocess
import platform
import requests
import zipfile
import shutil
from pathlib import Path

def check_ffmpeg():
    """Check if FFmpeg is already installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("FFmpeg is already installed and accessible")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    try:
        result = subprocess.run(['ffprobe', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("FFprobe is already installed and accessible")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return False

def install_ffmpeg_windows():
    """Install FFmpeg on Windows"""
    print("Installing FFmpeg for Windows...")
    
    # Create ffmpeg directory
    ffmpeg_dir = Path("ffmpeg")
    ffmpeg_dir.mkdir(exist_ok=True)
    
    # Download FFmpeg essentials build
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = ffmpeg_dir / "ffmpeg.zip"
    
    print("Downloading FFmpeg...")
    try:
        response = requests.get(ffmpeg_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Download completed")
        
        # Extract the zip file
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        # Find the extracted folder (it has a version number)
        extracted_folders = [f for f in ffmpeg_dir.iterdir() if f.is_dir()]
        if extracted_folders:
            ffmpeg_bin = extracted_folders[0] / "bin"
            
            # Copy executables to a known location
            local_bin = ffmpeg_dir / "bin"
            local_bin.mkdir(exist_ok=True)
            
            for exe in ["ffmpeg.exe", "ffprobe.exe", "ffplay.exe"]:
                src = ffmpeg_bin / exe
                dst = local_bin / exe
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"Copied {exe}")
            
            # Add to PATH for this session
            current_path = os.environ.get('PATH', '')
            new_path = str(local_bin.absolute())
            if new_path not in current_path:
                os.environ['PATH'] = f"{new_path};{current_path}"
                print(f"Added {new_path} to PATH")
            
            # Clean up
            os.remove(zip_path)
            shutil.rmtree(extracted_folders[0])
            
            print("FFmpeg installation completed!")
            return True
        else:
            print("Failed to find extracted FFmpeg folder")
            return False
            
    except Exception as e:
        print(f"Failed to install FFmpeg: {e}")
        return False

def install_with_conda():
    """Try to install FFmpeg with conda"""
    try:
        print("Trying to install FFmpeg with conda...")
        result = subprocess.run(['conda', 'install', '-c', 'conda-forge', 'ffmpeg', '-y'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("FFmpeg installed with conda")
            return True
        else:
            print(f"Conda installation failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Conda not found")
        return False

def install_with_chocolatey():
    """Try to install FFmpeg with Chocolatey"""
    try:
        print("Trying to install FFmpeg with Chocolatey...")
        result = subprocess.run(['choco', 'install', 'ffmpeg', '-y'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("FFmpeg installed with Chocolatey")
            return True
        else:
            print(f"Chocolatey installation failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Chocolatey not found")
        return False

def main():
    """Main setup function"""
    print("FFmpeg Setup for Accent Detector")
    print("=" * 40)
    
    # Check if already installed
    if check_ffmpeg():
        print("FFmpeg is already available. No setup needed!")
        return True
    
    print("FFmpeg not found. Installing...")
    
    if platform.system() == "Windows":
        # Try different installation methods
        methods = [
            ("Conda", install_with_conda),
            ("Chocolatey", install_with_chocolatey),
            ("Direct Download", install_ffmpeg_windows)
        ]
        
        for method_name, method_func in methods:
            print(f"\nTrying {method_name}...")
            if method_func():
                print(f"Successfully installed FFmpeg using {method_name}")
                
                # Verify installation
                if check_ffmpeg():
                    print("FFmpeg verification successful!")
                    return True
                else:
                    print("Installation completed but FFmpeg not accessible")
        
        print("\nAll installation methods failed.")
        print("\nManual Installation Instructions:")
        print("1. Download FFmpeg from: https://ffmpeg.org/download.html")
        print("2. Extract to a folder (e.g., C:\\ffmpeg)")
        print("3. Add C:\\ffmpeg\\bin to your system PATH")
        print("4. Restart your command prompt/IDE")
        
    else:
        print("For Linux/Mac, install FFmpeg using your package manager:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSetup completed successfully!")
        print("You can now run the accent detector app.")
    else:
        print("\nSetup incomplete. Please install FFmpeg manually.")
    
    input("\nPress Enter to exit...") 