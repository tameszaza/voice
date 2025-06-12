"""
Create 1×128×128 log-Mel tensors (npy) from wav files by slicing every
file into 4 s windows (last window zero-padded if needed, but
discarded if >65 % padding).

Run from project root:
$ python -m ResPreprocess.preprocess
Directory structure expected:
voice/
  ├─ database/   (contains *.wav)
  ├─ log-mel/    (will be created if absent)
  └─ ResPreprocess/
       └─ preprocess.py   (this file)
"""
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from skimage.transform import resize

# ----------------------- constants -----------------------
SR             = 16_000           # 16 kHz
SEGMENT_LEN    = 4 * SR          # 4 s segments
N_MELS         = 128
N_FFT          = 1024
HOP_LENGTH     = 512
TARGET_FRAMES  = 128              # time dimension after resize
MAX_PAD_RATIO  = 0.65             # discard if >65% padding in final chunk

# -------------------- helper functions -------------------
def load_audio(path: Path) -> np.ndarray:
    """Load a wav, convert to mono @ SR."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)

def wav_to_logmel(wav: np.ndarray) -> np.ndarray:
    """Compute log-Mel spectrogram (128 bands × T frames)."""
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # warp time axis to exactly TARGET_FRAMES
    if logmel.shape[1] != TARGET_FRAMES:
        logmel = resize(
            logmel,
            (N_MELS, TARGET_FRAMES),
            mode="reflect",
            preserve_range=True,
            anti_aliasing=True,
        )
    return logmel.astype(np.float32)

def save_tensor(tensor: np.ndarray, out_path: Path):
    """Save (1,128,128) tensor to .npy."""
    np.save(out_path, tensor)

# ------------------------ main ---------------------------
def main():
    wav_dir = Path("data_train/eval/fake")
    out_dir = Path("ResData/log-mel-eval/fake")
    out_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_dir.rglob("*.wav"):
        audio = load_audio(wav_path)
        total = len(audio)
        n_segs = math.ceil(total / SEGMENT_LEN)

        for idx in range(n_segs):
            start = idx * SEGMENT_LEN
            end   = start + SEGMENT_LEN
            chunk = audio[start:end]

            # if this is the final segment and it's shorter than SEGMENT_LEN,
            # compute padding ratio and possibly skip
            if len(chunk) < SEGMENT_LEN:
                pad_width = SEGMENT_LEN - len(chunk)
                pad_ratio = pad_width / SEGMENT_LEN
                if pad_ratio > MAX_PAD_RATIO:
                    print(f"⚠ Skipping {wav_path.name}[{idx}] — "
                          f"{pad_ratio:.1%} padding (> {MAX_PAD_RATIO:.0%})")
                    continue
                # otherwise zero-pad
                chunk = np.pad(chunk, (0, pad_width), mode="constant")

            # compute and save
            spec     = wav_to_logmel(chunk)[None, ...]  # → (1,128,128)
            out_name = f"{wav_path.stem}_{idx}.npy"
            save_tensor(spec, out_dir / out_name)
            print(f"✓ {wav_path.name}[{idx}] → {out_name}")

if __name__ == "__main__":
    main()
