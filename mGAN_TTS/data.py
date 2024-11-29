# process_audio.py

import os
import fnmatch
import numpy as np
import librosa
import random
import pickle
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

def find_files(path, pattern="*.flac"):
    """
    Recursively find all files matching the given pattern in the specified path.

    Args:
        path (str): The root directory to search for files.
        pattern (str): The glob pattern to match files. Default is '*.flac'.

    Returns:
        list: A list of file paths matching the pattern.
    """
    filenames = []
    for root, dirs, files in os.walk(path):
        matched_files = fnmatch.filter(files, pattern)
        for filename in matched_files:
            filenames.append(os.path.join(root, filename))
    return filenames

def load_flac(file_path, target_sr=16000):
    """
    Load a FLAC audio file and resample it to the target sample rate.

    Args:
        file_path (str): Path to the FLAC file.
        target_sr (int): Target sample rate. Default is 16000 Hz.

    Returns:
        numpy.ndarray: Normalized audio waveform.
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val  # Normalize to [-1, 1]
        else:
            audio = audio.astype(np.float32)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def pad_or_trim(audio, target_length=32000):
    """
    Pad or trim audio to the target length.

    Args:
        audio (numpy.ndarray): Audio waveform.
        target_length (int): Desired length of the audio waveform.

    Returns:
        numpy.ndarray: Padded or trimmed audio waveform.
    """
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    else:
        audio = audio[:target_length]
    return audio

def convert_audio(audio, sample_rate=16000, n_mels=128, hop_length=256):
    """
    Convert audio waveform to mel spectrogram.

    Args:
        audio (numpy.ndarray): Audio waveform.
        sample_rate (int): Sample rate of the audio.
        n_mels (int): Number of mel bands.
        hop_length (int): Number of samples between successive frames.

    Returns:
        tuple: (mel_spectrogram, audio)
    """
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, audio

def data_prepare(audio_path, mel_path, wav_file, target_sr=16000, target_length=32000):
    """
    Process a single audio file: load, pad/trim, convert to mel spectrogram,
    and save both audio and mel spectrogram as .npy files.

    Args:
        audio_path (str): Path to save the processed audio .npy file.
        mel_path (str): Path to save the mel spectrogram .npy file.
        wav_file (str): Path to the input audio file.
        target_sr (int): Target sample rate.
        target_length (int): Target length for the audio waveform.

    Returns:
        tuple: (audio_path, mel_path, mel_frames)
    """
    try:
        # Load and preprocess audio
        audio = load_flac(wav_file, target_sr)
        if audio is None:
            return None
        audio = pad_or_trim(audio, target_length)

        # Convert to mel spectrogram
        mel, _ = convert_audio(audio, sample_rate=target_sr)

        # Save audio and mel spectrogram as .npy files
        np.save(audio_path, audio, allow_pickle=False)
        np.save(mel_path, mel, allow_pickle=False)

        return audio_path, mel_path, mel.shape[1]  # Assuming mel.shape = (n_mels, time_steps)
    except Exception as e:
        print(f"Error processing file {wav_file}: {e}")
        return None

def process(output_dir, wav_files, train_dir, test_dir, num_workers, target_sr=16000, target_length=32000):
    """
    Process audio files in parallel and save processed data.

    Args:
        output_dir (str): Root output directory.
        wav_files (list): List of audio file paths.
        train_dir (str): Directory to save training data.
        test_dir (str): Directory to save testing data.
        num_workers (int): Number of parallel workers.
        target_sr (int): Target sample rate.
        target_length (int): Target length for the audio waveform.

    Returns:
        list: List of metadata tuples.
    """
    train_rate = 0.9  # Define train/test split ratio
    executor = ProcessPoolExecutor(max_workers=num_workers)
    results = []
    train_names = []
    test_names = []

    # Shuffle the wav_files list for random train/test split
    random.shuffle(wav_files)
    train_num = int(len(wav_files) * train_rate)

    # Process training files
    for wav_file in wav_files[:train_num]:
        fid = os.path.basename(wav_file).replace('.flac', '.npy')
        train_names.append(fid)
        audio_save_path = os.path.join(train_dir, "audio", fid)
        mel_save_path = os.path.join(train_dir, "mel", fid)
        results.append(executor.submit(
            data_prepare,
            audio_save_path,
            mel_save_path,
            wav_file,
            target_sr,
            target_length
        ))

    # Process testing files
    for wav_file in wav_files[train_num:]:
        fid = os.path.basename(wav_file).replace('.flac', '.npy')
        test_names.append(fid)
        audio_save_path = os.path.join(test_dir, "audio", fid)
        mel_save_path = os.path.join(test_dir, "mel", fid)
        results.append(executor.submit(
            data_prepare,
            audio_save_path,
            mel_save_path,
            wav_file,
            target_sr,
            target_length
        ))

    # Save training names
    with open(os.path.join(output_dir, "train", 'names.pkl'), 'wb') as f:
        pickle.dump(train_names, f)

    # Save testing names
    with open(os.path.join(output_dir, "test", 'names.pkl'), 'wb') as f:
        pickle.dump(test_names, f)

    # Wait for all processing to complete and gather metadata
    metadata = []
    for future in tqdm(results, desc="Processing audio files"):
        try:
            result = future.result()
            if result is not None:
                metadata.append(result)
        except Exception as e:
            print(f"Error in processing: {e}")

    return metadata

def preprocess(wav_dir, output, num_workers, target_sr=16000, target_length=32000):
    """
    Preprocess the dataset by organizing directories and processing files.

    Args:
        wav_dir (str): Directory containing input .flac files.
        output (str): Output directory for processed data.
        num_workers (int): Number of parallel workers.
        target_sr (int): Target sample rate.
        target_length (int): Target length for the audio waveform.

    Returns:
        None
    """
    train_dir = os.path.join(output, 'train')
    test_dir = os.path.join(output, 'test')

    # Create necessary directories
    os.makedirs(output, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "mel"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "mel"), exist_ok=True)

    # Find all .flac files in the wav_dir
    wav_files = find_files(wav_dir, pattern="*.flac")
    print(f"Found {len(wav_files)} .flac files.")

    if len(wav_files) == 0:
        print("No .flac files found. Please check the wav_dir path.")
        return

    # Process the files
    metadata = process(
        output_dir=output,
        wav_files=wav_files,
        train_dir=train_dir,
        test_dir=test_dir,
        num_workers=num_workers,
        target_sr=target_sr,
        target_length=target_length
    )

    # Write metadata
    write_metadata(metadata, output)

def write_metadata(metadata, out_dir):
    """
    Write metadata to a text file and print summary.

    Args:
        metadata (list): List of metadata tuples.
        out_dir (str): Output directory to save metadata.

    Returns:
        None
    """
    hop_length = 256  # Example value
    sample_rate = 16000  # Example value

    with open(os.path.join(out_dir, 'metadata.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')

    frames = sum([m[2] for m in metadata])
    frame_shift_ms = hop_length * 1000 / sample_rate
    hours = frames * frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))

if __name__ == '__main__':
    # Define parameters
    wav_dir = r"data/LibriSpeech/LibriSpeech/dev-clean"  # Replace with your .flac files directory
    output = "processed_data"  # Replace with your desired output directory
    num_workers = cpu_count()  # Utilize all available CPU cores
    target_sr = 16000  # Target sample rate
    target_length = 32000  # Target audio length (number of samples)

    # Call the preprocess function
    preprocess(wav_dir, output, num_workers, target_sr, target_length)
