import os
import torch
import numpy as np
import soundfile as sf
from IPython.display import Audio, display

def save_waveform_to_audio(waveform, sample_rate, filename):
    """
    Save a waveform to an audio file.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()

    # Ensure waveform is a 1D array
    if len(waveform.shape) > 1:
        waveform = np.squeeze(waveform)

    if waveform.ndim != 1:
        raise ValueError(f"Waveform must be 1D, but got shape {waveform.shape}")

    # Normalize waveform to be within [-1, 1]
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    # Debug: Print waveform info
    print(f"Waveform shape: {waveform.shape}")
    print(f"Waveform dtype: {waveform.dtype}")
    print(f"Waveform max value: {np.max(waveform)}")
    print(f"Waveform min value: {np.min(waveform)}")

    # Save the waveform
    sf.write(filename, waveform, sample_rate)

def verify_waveform_to_audio(root_dir, sample_rate=16000, target_length=32000, output_dir="verified_audio"):
    """
    Verify waveform-to-audio conversion using preprocessed dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Preprocess the dataset
    dataset = preprocess_dataset(root_dir, sample_rate, target_length)
    
    # Select the first waveform for verification
    waveform = dataset.tensors[0]
    
    # Save the waveform
    filename = os.path.join(output_dir, "example_waveform.wav")
    save_waveform_to_audio(waveform, sample_rate, filename)
    print(f"Waveform saved to {filename}")
    
    # Play the audio (if in a Jupyter Notebook)
    try:
        display(Audio(filename))
    except:
        print("Playback is not supported in this environment.")

# Test with sine wave
if __name__ == "__main__":
    try:
        # Generate a sine wave for testing
        sample_rate = 16000
        duration = 1.0  # seconds
        frequency = 440  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        test_waveform = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Save and debug
        save_waveform_to_audio(test_waveform, sample_rate, "verified_audio/test_sine_wave.wav")
        print("Sine wave saved successfully.")
    except Exception as e:
        print(f"Error: {e}")
