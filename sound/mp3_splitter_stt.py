import os
import sys
import re
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr

# --- 1. CONFIGURATION ---

# !! CRITICAL !!
# YOU MUST REPLACE THE NEXT TWO LINES WITH THE ACTUAL PATHS ON YOUR COMPUTER
# This is the only way to fix the "[WinError 2] The system cannot find the file specified" error.

FFMPEG_PATH = r"C:\FFmpeg\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\FFmpeg\bin\ffprobe.exe"

# --- End of required editing ---


# Set pydub's paths
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

# Quake II WAV requirements
FRAME_RATE = 22050
SAMPLE_WIDTH = 2
MAX_FILENAME_LENGTH = 50

# Silence detection parameters
MIN_SILENCE_LEN = 450  # CHANGED: Was 1000
SILENCE_THRESH = -40   # Silence threshold in dBFS
KEEP_SILENCE = 50      # CHANGED: Was 500

# Initialize the Recognizer
r = sr.Recognizer()

# --- 2. PRE-RUN CHECKS & PATH FIX ---
print("=" * 70)
print("--- SCRIPT DIAGNOSTICS ---")
print(f"Attempting to use FFmpeg from:  {FFMPEG_PATH}")
print(f"Attempting to use FFprobe from: {FFPROBE_PATH}")
print("=" * 70)

# Check if the user has updated the default paths
if "C:\\PATH\\TO\\YOUR" in FFMPEG_PATH or "C:\\PATH\\TO\\YOUR" in FFPROBE_PATH:
    print("!!! FATAL ERROR: You have not set the FFmpeg paths in the script. !!!")
    print("\nTo fix this:")
    print("  1. Download FFmpeg (from https://ffmpeg.org/download.html).")
    print("  2. Extract the ZIP file (e.g., to C:\\FFmpeg).")
    print("  3. Open this .py script and edit lines 12 and 13 to match the")
    print("     full path to your ffmpeg.exe and ffprobe.exe files.")
    print(r"     (e.g., C:\FFmpeg\bin\ffmpeg.exe)")
    print("=" * 70)
    sys.exit(1)

# Check if the files *actually exist* at the specified paths
if not os.path.exists(FFMPEG_PATH):
    print(f"!!! FATAL ERROR: The file does not exist at the specified FFMPEG_PATH:")
    print(f"    {FFMPEG_PATH}")
    print("    Please correct the path in Section 1 of the script.")
    print(r"    Make sure you are linking to the .exe file, not just the folder.")
    sys.exit(1)
if not os.path.exists(FFPROBE_PATH):
    print(f"!!! FATAL ERROR: The file does not exist at the specified FFPROBE_PATH:")
    print(f"    {FFPROBE_PATH}")
    print("    Please correct the path in Section 1 of the script.")
    print(r"    Make sure you are linking to the .exe file, not just the folder.")
    sys.exit(1)

# --- !! NEW FIX FOR [WinError 2] DLL LOADING !! ---
# This adds the executables' directory to the script's PATH
# so Windows can find the required .dll files (like avutil.dll).
try:
    ffmpeg_bin_path = os.path.dirname(FFMPEG_PATH)
    print(f"Temporarily adding {ffmpeg_bin_path} to script's PATH to find DLLs...")
    os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ["PATH"]
    print("PATH updated successfully for this session.")
except Exception as e:
    print(f"Warning: Could not modify PATH environment variable: {e}")
# --- End of new fix ---


# --- 3. COMMAND-LINE ARGUMENT PARSING ---

if len(sys.argv) < 3:
    print("Usage: python mp3_splitter_stt.py <source_dir> <output_dir>")
    sys.exit(1)

TARGET_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# Ensure directories exist
if not os.path.isdir(TARGET_DIR):
    print(f"Error: Source directory not found: {TARGET_DIR}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Source Directory: {TARGET_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

# --- 4. HELPER FUNCTIONS ---

def sanitize_text(text):
    """Converts a phrase into a clean, safe filename."""
    text = text.lower()
    text = text.replace(' ', '_')
    # Remove any non-alphanumeric characters (except underscores)
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text[:MAX_FILENAME_LENGTH].strip('_')

def process_mp3_file_with_stt(filepath):
    print(f"\nProcessing: {filepath}")
    filename_base = os.path.splitext(os.path.basename(filepath))[0]
    
    # CHANGED: Create a subdirectory for this specific MP3's chunks
    new_output_subdir = os.path.join(OUTPUT_DIR, filename_base)
    os.makedirs(new_output_subdir, exist_ok=True)
    
    temp_wav_path = None 
    
    try:
        # Load audio and set Quake II compatibility properties
        sound = (
            AudioSegment.from_mp3(filepath)
            .set_frame_rate(FRAME_RATE)
            .set_sample_width(SAMPLE_WIDTH)
            .set_channels(1)
        )
            
        audio_chunks = split_on_silence(
            sound,
            min_silence_len=MIN_SILENCE_LEN,
            silence_thresh=SILENCE_THRESH,
            keep_silence=KEEP_SILENCE
        )
        
        for i, chunk in enumerate(audio_chunks):
            # CHANGED: Temp file is now in the new subdirectory
            temp_wav_path = os.path.join(new_output_subdir, f"temp_chunk_{i}.wav")
            
            # 1. Export the chunk to a temporary WAV file for the SR library
            chunk.export(temp_wav_path, format="wav")
            
            transcript = ""
            try:
                # 2. Transcribe the audio chunk (requires internet)
                with sr.AudioFile(temp_wav_path) as source:
                    audio_data = r.record(source)
                
                # Using Google Speech Recognition API
                transcript = r.recognize_google(audio_data)
                
            except sr.UnknownValueError:
                transcript = f"unknown_speech_{i}"
                print(f"  Warning: Could not transcribe chunk {i}. Using default name.")
            except sr.RequestError as e:
                transcript = f"api_error_{i}"
                print(f"  Warning: Could not request results from Google SR service; {e}")

            # 3. Create a clean filename
            safe_name = sanitize_text(transcript)
            
            # Fallback to an index if the safe name is empty
            if not safe_name:
                safe_name = f"chunk_{i:03d}"
            
            # CHANGED: Filename is *only* the transcript
            output_filename = f"{safe_name}.wav"
            # CHANGED: Filepath is inside the new subdirectory
            output_filepath = os.path.join(new_output_subdir, output_filename)
            
            # 4. Rename the temporary file to the final, transcribed name
            os.rename(temp_wav_path, output_filepath)

            print(f"  Exported {os.path.join(filename_base, output_filename)}")

    except Exception as e:
        # The WinError 2 will be caught here if FFmpeg path is wrong
        print(f"!!! Error processing {filepath}: {e}")
        if isinstance(e, FileNotFoundError) or "[WinError 2]" in str(e):
                print("    ^-- This [WinError 2] *almost always* means your FFMPEG/FFPROBE paths")
                print("        in Section 1 of the script are incorrect, or the executables are corrupted.")
        
    finally:
        # Clean up the temporary file if it exists
        if temp_wav_path and os.path.exists(temp_wav_path): 
             os.remove(temp_wav_path)

# --- 5. MAIN EXECUTION ---

print("-" * 30)

# CHANGED: Clean up temp files from *all subdirectories*
print("Cleaning up old temporary files...")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        if f.startswith("temp_chunk_"):
            try:
                os.remove(os.path.join(root, f))
            except Exception as e:
                print(f"Warning: Could not remove old temp file {os.path.join(root, f)}: {e}")

# Process all MP3 files
print("Starting batch processing...")
for filename in os.listdir(TARGET_DIR):
    if filename.endswith(".mp3"):
        filepath = os.path.join(TARGET_DIR, filename)
        process_mp3_file_with_stt(filepath)

print("-" * 30)
print("Processing complete.")