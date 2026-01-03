# ==============================================================================
# STEP 1: SETUP AND INSTALLATION
# ==============================================================================
from google.colab import drive
import os
import subprocess
import re
import pandas as pd
from tqdm import tqdm

# Mount your Google Drive
drive.mount('/content/drive')

# Install yt-dlp, the tool for downloading from YouTube
!pip install -q yt-dlp

# --- Define the NEW output directory for the howl audio data ---
output_directory = "/content/drive/My Drive/DogAudioHowlData"
os.makedirs(output_directory, exist_ok=True)

print("Setup complete. Output directory is ready at:", output_directory)

# ==============================================================================
# STEP 2: DOWNLOAD AND PROCESS THE UNBALANCED "HOWL" AUDIO
# ==============================================================================

# --- Configuration ---
# The download limit is now set to 250.
DOWNLOAD_LIMIT = 250

# The machine ID for "Howl"
target_ids = ['/m/07qf0zm']

# --- Download the UNBALANCED segment information ---
print("\nDownloading AudioSet unbalanced train segment data...")
if not os.path.exists('unbalanced_train_segments.csv'):
    !wget -q http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv

# Load the segment data
print("Loading segment data into memory...")
segments_df = pd.read_csv(
    'unbalanced_train_segments.csv',
    header=None,
    sep=', ',
    engine='python',
    names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
    comment='#'
)

print(f"Total segments in unbalanced dataset: {len(segments_df)}")

# --- Filter the DataFrame to find rows containing the "Howl" ID ---
def contains_target_id(row_labels):
    if not isinstance(row_labels, str):
        return False
    return any(target_id in row_labels for target_id in target_ids)

filtered_df = segments_df[segments_df['positive_labels'].apply(contains_target_id)].copy()
print(f"Found {len(filtered_df)} segments labeled as 'Howl'.")

# --- Main Download Loop ---
print(f"\nStarting download of up to {DOWNLOAD_LIMIT or len(filtered_df)} clips...")
count = 0
for index, row in filtered_df.iterrows():
    if DOWNLOAD_LIMIT and count >= DOWNLOAD_LIMIT:
        print(f"Download limit of {DOWNLOAD_LIMIT} reached.")
        break

    yt_id = row['YTID']
    start_time = row['start_seconds']

    output_filename = os.path.join(output_directory, f"{yt_id}_{int(float(start_time))}.wav")

    if os.path.exists(output_filename):
        # print(f"Skipping {os.path.basename(output_filename)}, already exists.")
        count += 1
        continue

    try:
        print(f"Processing YouTube ID: {yt_id}...")
        temp_filename = f"/content/{yt_id}.wav"

        yt_dlp_command = [
            'yt-dlp', '-q', '--extract-audio', '--audio-format', 'wav',
            '-o', temp_filename, f'https://www.youtube.com/watch?v={yt_id}'
        ]
        subprocess.run(yt_dlp_command, check=True, capture_output=True, timeout=120)

        end_time = row['end_seconds']
        ffmpeg_command = [
            'ffmpeg', '-i', temp_filename, '-ss', str(start_time),
            '-to', str(end_time), '-c', 'copy', output_filename
        ]
        subprocess.run(ffmpeg_command, check=True, capture_output=True, timeout=60)

        os.remove(temp_filename)

        print(f"✅ Successfully saved {os.path.basename(output_filename)}")
        count += 1

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to process {yt_id}. Video may be private or deleted.")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    except subprocess.TimeoutExpired:
        print(f"❌ Timed out processing {yt_id}. Skipping.")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    except Exception as e:
        print(f"❌ An unexpected error occurred for {yt_id}: {e}")

print("\n--- Download process finished! ---")