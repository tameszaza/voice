import os
import torch
from torch.utils.data import TensorDataset
import librosa
import numpy as np
import glob

def librispeech_to_mel(root_dir, target_sr=16000, n_mels=64, n_fft=1024, hop_length=512, target_length=128):
    """
    Convert raw LibriSpeech audio files to Mel spectrograms and return a TensorDataset.
    """
    flac_files = glob.glob(os.path.join(root_dir, '**', '*.flac'), recursive=True)
    mel_spectrograms = []  # Initialize list to store Mel spectrograms

    if len(flac_files) == 0:
        raise ValueError(f"No FLAC files found in directory: {root_dir}")

    for file in flac_files:
        try:
            # Load the audio file
            audio, sr = librosa.load(file, sr=target_sr)
            # Normalize audio to range [-1, 1]
            audio = audio / np.max(np.abs(audio))
            
            # Convert to Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Ensure the Mel spectrogram has a fixed length
            if mel_spec_db.shape[1] < target_length:
                # Pad if shorter
                pad_width = target_length - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Trim if longer
                mel_spec_db = mel_spec_db[:, :target_length]

            # Add channel dimension and append to list
            mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)  # Add channel dim
            mel_spectrograms.append(mel_tensor)
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if len(mel_spectrograms) == 0:
        raise ValueError("No valid audio files were processed into Mel spectrograms.")

    # Stack into a single tensor
    mel_spectrograms = torch.stack(mel_spectrograms)  # Shape: [num_samples, 1, n_mels, target_length]
    return TensorDataset(mel_spectrograms)
