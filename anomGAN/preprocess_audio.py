import os
import numpy as np
import librosa
import soundfile as sf
from skimage.transform import resize # For resizing spectrograms if needed

# Monkey-patch for librosa compatibility with newer numpy
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'float'):
    np.float = float


SR = 16000                         # Sampling rate (Consistent)
FFT_DURATION_MS = 64               # Paper: FFT Size 64ms
N_FFT = int(SR * (FFT_DURATION_MS / 1000.0)) # Calculate N_FFT samples (e.g., 1024 for 64ms at 16kHz)

FRAME_SHIFT_MS = 32                # Paper: Frame Shift 32ms
HOP_LENGTH = int(SR * (FRAME_SHIFT_MS / 1000.0)) # Calculate HOP_LENGTH samples (e.g., 512 for 32ms at 16kHz)

N_MELS = 128                       # Paper: Number of Mel filters 128 (Crucial)
TARGET_DURATION_S = 10.0           # DCASE 2023 Task 2 uses 10s clips
                                   # If using different data, adjust this.
                                   # For 128 time frames with N_FFT=1024, HOP=512: 4.128s

# Target shape for the network input (if resizing is done in preprocessing)
# The paper implies an input size of 128x128 for the spectrograms fed to the network.
TARGET_SPECTROGRAM_SHAPE = (N_MELS, 128) # (n_mels, n_time_frames)

# For normalization: These should be calculated from your *entire training dataset*
# and then applied consistently. These are placeholders.
# Option 1: Global Min-Max for training set (then clip during test)
GLOBAL_TRAIN_MIN_DB = -80.0 # Example, calculate from your training data
GLOBAL_TRAIN_MAX_DB = 1.9073486328125e-06  # Example, calculate from your training data
# Option 2: Global Mean/Std for training set
GLOBAL_TRAIN_MEAN_DB = 18.111190795898438 # Example, calculate from your training data
GLOBAL_TRAIN_STD_DB = 15.0   # Example, calculate from your training data


def calculate_num_time_frames(duration_s, sr, hop_length, n_fft):
    """Calculates the number of time frames for a given duration."""
    total_samples = int(duration_s * sr)
    # Librosa's melspectrogram centers frames, so padding is effectively added.
    # The formula for frames with centered FFT is often ceil(total_samples / hop_length)
    # Or more precisely for librosa's default padding:
    # frames = 1 + int((total_samples - n_fft) / hop_length) if total_samples >= n_fft else 1
    # However, librosa.stft pads by default so that the t-th frame is centered at t * hop_length.
    # So, a simpler approximation that matches librosa's output shape for melspectrogram is:
    num_frames = int(np.floor(total_samples / hop_length)) + 1
    return num_frames

def preprocess_clip_to_log_mel(y, sr, n_fft, hop_length, n_mels):
    """
    Compute log Mel-spectrogram for a single audio clip.
    Normalization should be applied *after* this, based on global training set stats.
    """
    # 1) Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels,
                                       fmin=0, # Common practice
                                       fmax=sr//2) # Common practice
    # 2) Convert to log scale (dB)
    # Using power_to_db. Alternatively, 20 * log10(S + epsilon) or 10 * log10(S_power + epsilon)
    log_S = librosa.power_to_db(S, ref=np.max) # Or use a fixed reference like 1.0 if not comparing relatively within clip
    # log_S = librosa.amplitude_to_db(S, ref=np.max) # If S was amplitude spec

    return log_S

def normalize_spectrogram(log_S, method="global_zscore"):
    """ Normalizes a log Mel spectrogram. """
    if method == "global_minmax":
        # Ensure GLOBAL_TRAIN_MIN_DB and GLOBAL_TRAIN_MAX_DB are defined
        # This scales to [0,1] based on training set's global min/max
        norm_S = (log_S - GLOBAL_TRAIN_MIN_DB) / (GLOBAL_TRAIN_MAX_DB - GLOBAL_TRAIN_MIN_DB + 1e-6)
        norm_S = np.clip(norm_S, 0, 1) # Clip to [0,1] range
        # For Tanh output in GAN, scale to [-1, 1]
        norm_S = norm_S * 2.0 - 1.0
        #raise NotImplementedError("Define GLOBAL_TRAIN_MIN_DB and GLOBAL_TRAIN_MAX_DB first.")
    elif method == "global_zscore":
        # Ensure GLOBAL_TRAIN_MEAN_DB and GLOBAL_TRAIN_STD_DB are defined
        norm_S = (log_S - GLOBAL_TRAIN_MEAN_DB) / (GLOBAL_TRAIN_STD_DB + 1e-6)
    elif method == "instance_minmax_to_0_1": # Your original method
        S_db_min, S_db_max = log_S.min(), log_S.max()
        norm_S = (log_S - S_db_min) / (S_db_max - S_db_min + 1e-6)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return norm_S


def process_directory_to_log_mel(input_dir_path, output_dir_path,
                                 sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 n_mels=N_MELS, target_duration_s=TARGET_DURATION_S,
                                 target_spec_shape=TARGET_SPECTROGRAM_SHAPE,
                                 normalization_method="global_zscore"):
    """
    Load .wav files, process into fixed-duration clips (or use full clip),
    compute log Mel-spectrograms, normalize, optionally resize, and save as .npy.
    """
    os.makedirs(output_dir_path, exist_ok=True)
    target_samples = int(sr * target_duration_s)

    print(f"Processing directory: {input_dir_path}")
    print(f"Parameters: SR={sr}, N_FFT={n_fft}, HOP_LENGTH={hop_length}, N_MELS={n_mels}")
    print(f"Target clip duration: {target_duration_s}s, Target samples per clip: {target_samples}")
    num_expected_frames_original_duration = calculate_num_time_frames(target_duration_s, sr, hop_length, n_fft)
    print(f"Spectrograms will have {n_mels} Mel bands.")
    print(f"Original number of time frames for {target_duration_s}s clip: {num_expected_frames_original_duration}")
    if target_spec_shape:
        print(f"Spectrograms will be resized to: {target_spec_shape}")


    for fname in os.listdir(input_dir_path):
        if not fname.lower().endswith(('.wav', '.flac')): # Support flac too
            continue
        
        path = os.path.join(input_dir_path, fname)
        base_name = os.path.splitext(fname)[0]

        try:
            y, current_sr = librosa.load(path, sr=sr, res_type="kaiser_best")
            if current_sr != sr:
                print(f"Warning: Resampled {fname} from {current_sr}Hz to {sr}Hz.")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # If using DCASE data, clips are already ~10s.
        # If your data is longer, you might want to segment it.
        # For simplicity, this version processes the whole clip (up to target_duration_s) or pads.

        if len(y) > target_samples:
            # If you want to strictly use DCASE 10s, you can truncate or take the first 10s.
            # For anomaly detection, often the start of a sound event is important.
            print(f"Clip {fname} is longer ({len(y)/sr:.2f}s) than target ({target_duration_s}s), truncating.")
            y = y[:target_samples]
        elif len(y) < target_samples:
            # Pad if shorter
            print(f"Clip {fname} is shorter ({len(y)/sr:.2f}s) than target ({target_duration_s}s), padding.")
            padding = target_samples - len(y)
            y = np.pad(y, (0, padding), 'constant')

        log_S = preprocess_clip_to_log_mel(y, sr, n_fft, hop_length, n_mels)

        # --- Normalization ---
        # Apply normalization based on pre-calculated global stats from the training set
        # This is crucial for consistent input to the GAN
        norm_log_S = normalize_spectrogram(log_S, method=normalization_method)

        # --- Optional: Resize spectrogram to fixed shape (e.g., 128x128) ---
        # The paper implies isize=128, so (128 Mels, 128 time_frames)
        if target_spec_shape:
            if norm_log_S.shape != target_spec_shape:
                # skimage.transform.resize output range is [0,1] by default if input is not bool.
                # If your normalization made it [-1,1] or Z-score, this needs care.
                # It's often better to resize *before* Z-score normalization if std dev changes much with size.
                # Or, ensure resize preserves the range.
                # For Z-score, resizing might slightly alter the distribution, but often acceptable.
                # Let's assume norm_log_S is the data we want to resize.
                # The output of resize is float64 by default, convert to float32
                resized_norm_log_S = resize(norm_log_S, target_spec_shape,
                                            anti_aliasing=True, mode='reflect', preserve_range=True)
                resized_norm_log_S = resized_norm_log_S.astype(np.float32)
                feature_to_save = resized_norm_log_S
            else:
                feature_to_save = norm_log_S.astype(np.float32)
        else:
            feature_to_save = norm_log_S.astype(np.float32)


        output_file_path = os.path.join(output_dir_path, f"{base_name}_logmel.npy")
        np.save(output_file_path, feature_to_save)

    print(f"Log Mel spectrogram processing complete. Saved features to: {output_dir_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess .wav files into log Mel spectrograms.")
    parser.add_argument("input_directory", help="Path to folder with .wav or .flac files")
    parser.add_argument("output_directory", help="Folder to save generated .npy log Mel spectrograms")
    
    parser.add_argument("--sr", type=int, default=SR, help="Target sampling rate")
    parser.add_argument("--n_fft_ms", type=float, default=FFT_DURATION_MS, help="FFT window duration in ms")
    parser.add_argument("--hop_length_ms", type=float, default=FRAME_SHIFT_MS, help="Hop length (frame shift) in ms")
    parser.add_argument("--n_mels", type=int, default=N_MELS, help="Number of Mel filter banks")
    parser.add_argument("--duration_s", type=float, default=TARGET_DURATION_S, help="Target clip duration in seconds for processing")
    
    parser.add_argument("--target_rows", type=int, default=TARGET_SPECTROGRAM_SHAPE[0], help="Target rows (Mel bands) for output spectrogram after resize")
    parser.add_argument("--target_cols", type=int, default=TARGET_SPECTROGRAM_SHAPE[1], help="Target columns (time frames) for output spectrogram after resize")
    parser.add_argument("--no_resize", action="store_true", help="Do not resize the spectrogram to target_rows/cols")

    parser.add_argument("--norm_method", type=str, default="global_zscore",
                        choices=["global_zscore", "global_minmax", "instance_minmax_to_0_1"],
                        help="Normalization method. For global methods, ensure you've set the constants.")

    args = parser.parse_args()

    # Recalculate N_FFT and HOP_LENGTH from ms if provided via args
    current_n_fft = int(args.sr * (args.n_fft_ms / 1000.0))
    current_hop_length = int(args.sr * (args.hop_length_ms / 1000.0))
    
    current_target_spec_shape = None
    if not args.no_resize:
        current_target_spec_shape = (args.target_rows, args.target_cols)
        if args.target_rows != args.n_mels:
            print(f"Warning: --target_rows ({args.target_rows}) is different from --n_mels ({args.n_mels}). Mel bands will be {args.n_mels} before resize.")


    # --- CRITICAL ---
    # Before running, you MUST calculate GLOBAL_TRAIN_MEAN_DB and GLOBAL_TRAIN_STD_DB
    # (or MIN/MAX) from your *entire training dataset* if using global normalization.
    # Example:
    if args.norm_method == "global_zscore":
        print(f"Using global Z-score normalization with MEAN={GLOBAL_TRAIN_MEAN_DB}, STD={GLOBAL_TRAIN_STD_DB}")
        print("Ensure these global constants are correctly calculated from your training set!")
    elif args.norm_method == "global_minmax":
        print(f"Using global Min-Max normalization with MIN={GLOBAL_TRAIN_MIN_DB}, MAX={GLOBAL_TRAIN_MAX_DB}")
        print("Ensure these global constants are correctly calculated from your training set!")


    process_directory_to_log_mel(
        args.input_directory,
        args.output_directory,
        sr=args.sr,
        n_fft=current_n_fft,
        hop_length=current_hop_length,
        n_mels=args.n_mels,
        target_duration_s=args.duration_s,
        target_spec_shape=current_target_spec_shape,
        normalization_method=args.norm_method
    )