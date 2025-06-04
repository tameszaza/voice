import os
import numpy as np
import librosa
# Import your preprocessing functions: calculate_num_time_frames, preprocess_clip_to_log_mel
# from your_preprocessing_script import preprocess_clip_to_log_mel, SR, N_FFT, HOP_LENGTH, N_MELS, TARGET_DURATION_S

# --- Define these based on your_preprocessing_script.py ---
SR_CALC = 16000
FFT_DURATION_MS_CALC = 64
N_FFT_CALC = int(SR_CALC * (FFT_DURATION_MS_CALC / 1000.0))
FRAME_SHIFT_MS_CALC = 32
HOP_LENGTH_CALC = int(SR_CALC * (FRAME_SHIFT_MS_CALC / 1000.0))
N_MELS_CALC = 128
TARGET_DURATION_S_CALC = 10.0 # Or whatever duration you'll use for actual processing
# -------------------------------------------------------------

def preprocess_clip_to_log_mel_for_stats(y, sr, n_fft, hop_length, n_mels): # Simplified from your main script
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=sr//2)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

def calculate_global_stats(input_dir_path, sr, n_fft, hop_length, n_mels, target_duration_s):
    all_log_mels = []
    target_samples = int(sr * target_duration_s)
    print(f"Calculating global stats from: {input_dir_path}")

    for fname in os.listdir(input_dir_path):
        if not fname.lower().endswith(('.wav', '.flac')):
            continue
        path = os.path.join(input_dir_path, fname)
        try:
            y, current_sr = librosa.load(path, sr=sr, res_type="kaiser_best")
            if len(y) > target_samples:
                y = y[:target_samples]
            elif len(y) < target_samples:
                padding = target_samples - len(y)
                y = np.pad(y, (0, padding), 'constant')

            log_S = preprocess_clip_to_log_mel_for_stats(y, sr, n_fft, hop_length, n_mels)
            all_log_mels.append(log_S)
        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")

    if not all_log_mels:
        print("No spectrograms processed. Cannot calculate stats.")
        return None, None, None, None

    # Stack all spectrograms to calculate global mean/std
    # This can be memory intensive if you have a huge dataset.
    # If so, calculate mean/std iteratively (more complex).
    try:
        stacked_log_mels = np.stack(all_log_mels)
        global_mean = np.mean(stacked_log_mels)
        global_std = np.std(stacked_log_mels)
        global_min = np.min(stacked_log_mels)
        global_max = np.max(stacked_log_mels)
        print(f"--- Global Stats for Training Data ---")
        print(f"Calculated Global Mean (dB): {global_mean}")
        print(f"Calculated Global Std Dev (dB): {global_std}")
        print(f"Calculated Global Min (dB): {global_min}")
        print(f"Calculated Global Max (dB): {global_max}")
        print(f"--------------------------------------")
        return global_mean, global_std, global_min, global_max
    except ValueError as ve:
        print(f"Error stacking spectrograms (likely due to inconsistent shapes before resize): {ve}")
        print("Ensure all clips produce spectrograms of the same dimension before stacking for stats, or resize them first.")
        print("Shapes found:")
        for i, spec in enumerate(all_log_mels):
            print(f"Spec {i}: {spec.shape}")
            if i > 5: break # Print a few
        return None, None, None, None


# --- Run this once for your training data ---
training_audio_dir = "voice/data_wavefake/real/wavs" # YOUR TRAINING AUDIO PATH
# These parameters should match what you'll use for actual preprocessing
g_mean, g_std, g_min, g_max = calculate_global_stats(
    training_audio_dir,
    sr=SR_CALC,
    n_fft=N_FFT_CALC,
    hop_length=HOP_LENGTH_CALC,
    n_mels=N_MELS_CALC,
    target_duration_s=TARGET_DURATION_S_CALC
)