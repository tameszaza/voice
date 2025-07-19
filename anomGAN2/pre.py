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

def process_directory(input_root,
                      output_root,
                      sr=SR,
                      n_fft=N_FFT,
                      hop_length=HOP_LENGTH,
                      n_mels=DEFAULT_N_MELS,
                      duration=DEFAULT_DURATION):
    """
    Recursively process all subfolders containing .wav files.
    For each folder, compute mel & mfcc for every fixed‐length clip,
    stack them in memory, then write a single stacked.npy per channel.
    """
    target_samples = int(sr * duration)

    for root, dirs, files in os.walk(input_root):
        wavs = [f for f in files if f.lower().endswith('.wav')]
        if not wavs:
            continue

        # build output subfolders
        rel = os.path.relpath(root, input_root)
        mel_out  = os.path.join(output_root, rel, 'mel')
        mfcc_out = os.path.join(output_root, rel, 'mfcc')
        os.makedirs(mel_out,  exist_ok=True)
        os.makedirs(mfcc_out, exist_ok=True)

        # collect all clips in memory
        mel_clips  = []
        mfcc_clips = []

        for wav in wavs:
            path = os.path.join(root, wav)
            y, _ = librosa.load(path, sr=sr, res_type='kaiser_best')
            y_trim, _ = librosa.effects.trim(y)

            # how many full-duration clips
            n_full = len(y_trim) // target_samples
            for i in range(n_full):
                clip = y_trim[i*target_samples:(i+1)*target_samples]
                S, M = preprocess_clip(clip, sr, n_fft, hop_length, n_mels)
                print(f"  • {wav}: clip {i+1}/{n_full} processed, shape {S.shape}")
                mel_clips.append(S)
                mfcc_clips.append(M)

        # helper to stack & save one file per folder
        def stack_and_save(arr_list, folder):
            if not arr_list:
                print(f"  • no clips in {folder}, skipping")
                return
            ref_shape = arr_list[0].shape
            for arr in arr_list:
                if arr.shape != ref_shape:
                    raise ValueError(f"Shape mismatch in {folder}: {arr.shape} vs {ref_shape}")
            stacked = np.stack(arr_list, axis=0)  # shape (N, H, W)
            out_path = os.path.join(folder, 'stacked.npy')
            np.save(out_path, stacked)
            print(f"  • {folder}: saved stacked.npy with shape {stacked.shape}")

        # replace per‐clip .npy files with these two stacked outputs
        stack_and_save(mel_clips,  mel_out)
        stack_and_save(mfcc_clips, mfcc_out)


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
    parser.add_argument("output_dir", help="Folder to save generated .npy files (ignored, output will be placed in input subfolders)")
    parser.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS,
                        help="Number of Mel filter banks")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="Target clip duration in seconds")

    args = parser.parse_args()
    process_directory(args.directory, args.output_dir,
                      n_mels=args.n_mels, duration=args.duration)

    print("Preprocessing complete. Saved features to subfolders of:")