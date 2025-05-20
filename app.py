import os
import streamlit as st
import requests
from pytube import YouTube
from pathlib import Path
import logging
import traceback
from datetime import datetime
import subprocess
import sys
import tempfile
import shutil
import re
from moviepy.editor import VideoFileClip

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

# App header
st.title("Video Downloader")
st.markdown("Download videos from Loom, direct MP4 links, and more")

# URL input
url = st.text_input("Enter the URL of the video to download:", placeholder="https://www.loom.com/... or direct MP4 URL")

# Audio extraction option
extract_audio = st.checkbox("Extract audio to WAV format", value=False)

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

# Download button
if st.button("Download"):
    if not url:
        st.error("Please enter a valid URL")
        logger.warning("Download attempted with empty URL")
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
                
                # Extract audio if option selected and file downloaded successfully
                if extract_audio and downloaded_file_path:
                    st.info("Extracting audio from video...")
                    audio_path = extract_audio_from_video(downloaded_file_path)
                    if audio_path:
                        st.success(f"Audio extracted successfully: {audio_path.name}")
                    else:
                        st.error("Failed to extract audio from video")
                
                # Display downloaded files
                st.subheader("Downloaded Files")
                files = list(download_dir.glob("*"))
                logger.info(f"Files in download directory: {[f.name for f in files]}")
                for file in files:
                    st.write(f"📁 {file.name}")
                
                # Display extracted audio files
                if extract_audio:
                    st.subheader("Extracted Audio Files")
                    audio_files = list(audio_dir.glob("*.wav"))
                    logger.info(f"Audio files in directory: {[f.name for f in audio_files]}")
                    for file in audio_files:
                        st.write(f"🔊 {file.name}")
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Unexpected error: {str(e)}\n{error_details}")
            st.error(f"Error: {str(e)}")

# Display download folder info
st.markdown("---")
st.info(f"Files are saved to: {os.path.abspath(download_dir)}")
if extract_audio:
    st.info(f"Audio files are saved to: {os.path.abspath(audio_dir)}")

# Add tips about Loom and direct links
st.markdown("""
### Tips for Getting Video URLs
- **Loom Videos**: Right-click on a Loom video and select "Copy Video Address" to get a direct URL
- **Direct MP4 Links**: Look for URLs ending with .mp4, .mov, etc.
- **Other Video Sites**: Right-click on the video and look for options like "Copy video address"
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
    """)

logger.info("App session ended") 