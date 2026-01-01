"""
Augmenting aggressive dog audios

Single-folder audio augmenter
- Put original .wav files in INPUT_DIR
- Augmented files saved to OUTPUT_DIR
- Conservative transforms to preserve dog vocal identity

Dependencies (install once):
pip install librosa soundfile scipy numpy
"""

import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve

# ------------------ REPRODUCIBILITY ------------------
random.seed(42)
np.random.seed(42)

# ------------------ USER CONFIG ------------------
INPUT_DIR = "data/aggressive/original"     # folder containing .wav files
OUTPUT_DIR = "data/aggressive/augmented"   # output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 32000                  # sample rate
TARGET_PER_FILE_AUGS = 5    # augmentations per file
MIN_DURATION_SEC = 0.5
MAX_AMPSCALE = 0.98
# ---------------------------------------------------


# ================== UTILITY FUNCTIONS ==================
def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def save_audio(path, y, sr=SR):
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak * MAX_AMPSCALE
    sf.write(path, y.astype(np.float32), sr)


def pad_or_trim(y, target_len):
    if len(y) < target_len:
        pad_len = target_len - len(y)
        left = pad_len // 2
        right = pad_len - left
        y = np.pad(y, (left, right), mode="constant")
    else:
        y = y[:target_len]
    return y


def rms(y):
    return np.sqrt(np.mean(y ** 2) + 1e-16)


# ================== AUGMENTATION OPS ==================
def time_stretch_small(y):
    rate = random.uniform(0.9, 1.1)
    try:
        return librosa.effects.time_stretch(y, rate)
    except Exception:
        return y


def pitch_shift_small(y, sr):
    steps = random.uniform(-1.5, 1.5)
    try:
        return librosa.effects.pitch_shift(y, sr, steps)
    except Exception:
        return y


def add_white_noise(y, snr_db=15.0):
    r = rms(y)
    noise_rms = r / (10 ** (snr_db / 20.0))
    noise = np.random.normal(0, noise_rms, size=y.shape)
    return y + noise


def add_pink_noise(y, snr_db=15.0):
    wn = np.random.randn(len(y))
    pn = np.cumsum(wn)
    pn = pn / (np.max(np.abs(pn)) + 1e-12)
    r = rms(y)
    noise_rms = r / (10 ** (snr_db / 20.0))
    return y + pn * noise_rms


def random_shift(y, max_shift_sec=0.15, sr=SR):
    max_shift = int(max_shift_sec * sr)
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        y = np.pad(y, (shift, 0), mode="constant")[:len(y)]
    elif shift < 0:
        y = np.pad(y, (0, -shift), mode="constant")[-shift:len(y) - shift]
    return y


def change_gain(y, db):
    gain = 10 ** (db / 20.0)
    return y * gain


def highpass(y, sr, cutoff=120.0, order=4):
    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, y)


def simple_reverb(y, sr, reverb_seconds=0.08, decay=4.0):
    ir_len = int(reverb_seconds * sr)
    if ir_len < 1:
        return y
    t = np.linspace(0, reverb_seconds, ir_len)
    ir = np.exp(-decay * t)
    ir = ir / (np.sum(np.abs(ir)) + 1e-12)
    rev = fftconvolve(y, ir, mode="full")[:len(y)]
    return 0.85 * y + 0.15 * rev


def augment_once(y, sr=SR):
    out = y.copy()
    ops_count = random.randint(1, 3)
    candidates = [
        "stretch", "pitch", "white_noise", "pink_noise",
        "shift", "gain", "highpass", "reverb"
    ]
    chosen_ops = random.sample(candidates, k=ops_count)

    for op in chosen_ops:
        if op == "stretch":
            out = time_stretch_small(out)
        elif op == "pitch":
            out = pitch_shift_small(out, sr)
        elif op == "white_noise":
            out = add_white_noise(out, snr_db=random.uniform(10, 20))
        elif op == "pink_noise":
            out = add_pink_noise(out, snr_db=random.uniform(10, 18))
        elif op == "shift":
            out = random_shift(out, max_shift_sec=0.12, sr=sr)
        elif op == "gain":
            out = change_gain(out, db=random.uniform(-6, 6))
        elif op == "highpass":
            out = highpass(out, sr, cutoff=random.uniform(80, 300))
        elif op == "reverb":
            out = simple_reverb(
                out,
                sr,
                reverb_seconds=random.uniform(0.03, 0.18),
                decay=random.uniform(2.0, 6.0),
            )

    return out, chosen_ops


# ================== MAIN SCRIPT ==================
def main():
    src_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.wav")))
    if not src_files:
        raise RuntimeError(f"No .wav files found in {INPUT_DIR}")

    print(f"Found {len(src_files)} files.")
    print(f"Generating {TARGET_PER_FILE_AUGS} augmentations per file...\n")

    for src in src_files:
        base = os.path.splitext(os.path.basename(src))[0]
        y = load_audio(src, sr=SR)
        if len(y) == 0:
            continue

        target_len = int(SR * max(MIN_DURATION_SEC, len(y) / SR))

        save_audio(
            os.path.join(OUTPUT_DIR, f"{base}_orig.wav"),
            pad_or_trim(y, target_len),
            SR,
        )

        for k in range(TARGET_PER_FILE_AUGS):
            aug_y, _ = augment_once(y, sr=SR)
            aug_y = pad_or_trim(aug_y, target_len)
            save_audio(
                os.path.join(OUTPUT_DIR, f"{base}_aug{k:03d}.wav"),
                aug_y,
                SR,
            )

        print(f"✔ Augmented: {base}")

    print("\nDone. Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
