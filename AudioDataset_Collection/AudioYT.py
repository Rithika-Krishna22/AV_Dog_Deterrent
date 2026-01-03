import subprocess
import re
import pandas as pd
import os

# --- EDIT THIS LINE ---
# Option 1: To download all remaining clips, set to None
DOWNLOAD_LIMIT = None

# Option 2: To download 50 more (for a total of 100), set to 100
# DOWNLOAD_LIMIT = 100
# --------------------

output_directory = "/content/drive/My Drive/DogAudioDataset"
target_ids = ['/m/0bt9lr', '/m/05tny_', '/m/0ghcn6']

print("Loading AudioSet segment data...")
segments_df = pd.read_csv(
    'balanced_train_segments.csv',
    header=None,
    sep=', ',
    engine='python',
    names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
    comment='#'
)

def contains_target_id(row_labels):
    if not isinstance(row_labels, str):
        return False
    return any(target_id in row_labels for target_id in target_ids)

filtered_df = segments_df[segments_df['positive_labels'].apply(contains_target_id)].copy()
print(f"Found {len(filtered_df)} total segments matching your labels.")

print(f"\nResuming download...")
count = 0
for index, row in filtered_df.iterrows():
    if DOWNLOAD_LIMIT and count >= DOWNLOAD_LIMIT:
        print(f"Download limit of {DOWNLOAD_LIMIT} reached.")
        break

    yt_id = row['YTID']
    start_time = row['start_seconds']

    output_filename = os.path.join(output_directory, f"{yt_id}_{int(float(start_time))}.wav")

    if os.path.exists(output_filename):
        # This is the check that skips already downloaded files
        # print(f"Skipping {os.path.basename(output_filename)}, already exists.")
        count += 1 # We still increment the count to respect the new limit
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