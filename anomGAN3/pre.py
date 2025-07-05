import os
import numpy as np
import librosa
import soundfile as sf  # Put this import at the top of the file

# Monkey-patch for librosa compatibility with newer numpy
np.complex = complex

# === Parameters (adjust as needed) ===
SR = 16000               # Sampling rate
N_FFT = 2048             # FFT window size
HOP_LENGTH = 512         # Hop length between frames
DEFAULT_N_MELS = 32     # Default number of Mel filters
DEFAULT_DURATION = 3.0   # Target clip duration in seconds

def preprocess_clip(y, sr, n_fft, hop_length, n_mels):
    """
    Compute Mel-spectrogram and normalized MFCC for a single audio clip.
    Both outputs are normalized to [0, 1].
    """
    # 1) Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize Mel-spectrogram to [0, 1]
    S_db_min, S_db_max = S_db.min(), S_db.max()
    S_db_norm = (S_db - S_db_min) / (S_db_max - S_db_min + 1e-6)

    # 2) MFCC (from Mel-spectrogram)
    mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=n_mels)

    # 3) Normalize MFCC to [0, 1]
    mfcc_min, mfcc_max = mfcc.min(), mfcc.max()
    mfcc_norm = (mfcc - mfcc_min) / (mfcc_max - mfcc_min + 1e-6)

    return S_db_norm, mfcc_norm

def process_directory(dir_path, output_dir,
                      sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      n_mels=DEFAULT_N_MELS, target_duration=DEFAULT_DURATION):
    """
    Recursively process all subfolders containing .wav files.
    Save features in the same relative subfolder structure under output_dir.
    """
    for root, dirs, files in os.walk(dir_path):
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        if not wav_files:
            continue  # Skip folders without wav files

        # Compute relative path from input root to current folder
        rel_path = os.path.relpath(root, dir_path)
        mel_out = os.path.join(output_dir, rel_path, 'mel')
        mfcc_out = os.path.join(output_dir, rel_path, 'mfcc')
        os.makedirs(mel_out, exist_ok=True)
        os.makedirs(mfcc_out, exist_ok=True)

        target_samples = int(sr * target_duration)
        leftovers = []

        for fname in wav_files:
            path = os.path.join(root, fname)
            base = os.path.splitext(fname)[0]

            # Load & trim silence
            y, _ = librosa.load(path, sr=sr, res_type="kaiser_best")
            y_trim, _ = librosa.effects.trim(y)

            # Chop into full-length clips
            n_full = len(y_trim) // target_samples
            # Save comparison samples in the same relative structure (optional)
            # comparison_dir = os.path.join(output_dir, rel_path, "comparison_samples")
            # os.makedirs(comparison_dir, exist_ok=True)

            for i in range(n_full):
                start = i * target_samples
                end = (i + 1) * target_samples
                clip = y_trim[start:end]

                # Save original (non-resampled) if within first 5 clips
                # if i < 5:
                #     y_original, original_sr = librosa.load(path, sr=None)
                #     original_clip = y_original[int(start * (original_sr / sr)) : int(end * (original_sr / sr))]
                #     sf.write(os.path.join(comparison_dir, f"{base}_orig_clip{i}.wav"), original_clip, original_sr)
                #     sf.write(os.path.join(comparison_dir, f"{base}_down_clip{i}.wav"), clip, sr)

                S_db, mfcc_norm = preprocess_clip(clip, sr, n_fft, hop_length, n_mels)
                np.save(os.path.join(mel_out, f"{base}_clip{i}.npy"), S_db)
                np.save(os.path.join(mfcc_out, f"{base}_clip{i}.npy"), mfcc_norm)

            # Collect leftover
            rem = y_trim[n_full*target_samples:]
            if rem.size > 0:
                leftovers.append((base, rem))

        # Concatenate and process leftovers (optional, still commented out)
        # if leftovers:
        #     # Combine residuals into one long buffer
        #     all_res = np.concatenate([seg for _, seg in leftovers])
        #     n_full = len(all_res) // target_samples

        #     for i in range(n_full):
        #         clip = all_res[i*target_samples:(i+1)*target_samples]
        #         S_db, mfcc_norm = preprocess_clip(clip, sr, n_fft, hop_length, n_mels)

        #         np.save(os.path.join(mel_out, f"leftover_clip{i}.npy"), S_db)
        #         np.save(os.path.join(mfcc_out, f"leftover_clip{i}.npy"), mfcc_norm)

        #     final_rem = all_res[n_full*target_samples:]
        #     if final_rem.size > 0:
        #         # Pad final segment
        #         pad = target_samples - final_rem.size
        #         padded = np.pad(final_rem, (0, pad))

        #         S_db, mfcc_norm = preprocess_clip(padded, sr, n_fft, hop_length, n_mels)
        #         np.save(os.path.join(mel_out, f"leftover_final.npy"), S_db)
        #         np.save(os.path.join(mfcc_out, f"leftover_final.npy"), mfcc_norm)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess .wav files "
                                                 "and save features")
    parser.add_argument("directory", help="Path to folder with .wav files")
    parser.add_argument("output_dir", help="Folder to save generated .npy files")
    parser.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS,
                        help="Number of Mel filter banks")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="Target clip duration in seconds")

    args = parser.parse_args()
    process_directory(args.directory, args.output_dir,
                      n_mels=args.n_mels, target_duration=args.duration)

    print("Preprocessing complete. Saved features to:", args.output_dir)
