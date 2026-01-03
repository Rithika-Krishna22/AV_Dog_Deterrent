import subprocess
import re
import pandas as pd
import os

# --- Configuration ---
DOWNLOAD_LIMIT = 50
output_directory = "/content/drive/My Drive/DogAudioDataset"
target_ids = ['/m/0bt9lr', '/m/05tny_', '/m/0ghcn6']

# Download the segment information
print("Downloading AudioSet segment data...")
if not os.path.exists('balanced_train_segments.csv'):
    !wget -q http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv

# --- THIS IS THE CORRECTED LINE ---
# Load the segment data, ignoring lines that start with '#'
segments_df = pd.read_csv(
    'balanced_train_segments.csv',
    header=None,
    sep=', ',
    engine='python',
    names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
    comment='#' # This tells pandas to ignore header/comment lines
)

print(f"Total segments in dataset: {len(segments_df)}")

# --- The robust search function ---
def contains_target_id(row_labels):
    if not isinstance(row_labels, str):
        return False
    return any(target_id in row_labels for target_id in target_ids)

filtered_df = segments_df[segments_df['positive_labels'].apply(contains_target_id)].copy()
print(f"Found {len(filtered_df)} segments matching your labels.")

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
        print(f"Skipping {output_filename}, already exists.")
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