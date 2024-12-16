import os
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

def convert_single_file(flac_path, wav_path):
    """
    Converts a single .flac file to .wav format.
    """
    try:
        audio = AudioSegment.from_file(flac_path, format="flac")
        audio.export(wav_path, format="wav")
        print(f"Converted: {flac_path} -> {wav_path}")
    except Exception as e:
        print(f"Failed to convert {flac_path}: {e}")

def convert_flac_to_wav(input_dir, output_dir, num_workers=4):
    """
    Converts all .flac files in a given directory and its subdirectories to .wav files,
    using multiple threads for faster processing.

    :param input_dir: Root directory containing .flac files.
    :param output_dir: Directory to save the converted .wav files.
    :param num_workers: Number of threads to use.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all .flac files and their output paths
    files_to_convert = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                flac_path = os.path.join(root, file)
                wav_filename = os.path.splitext(file)[0] + ".wav"
                wav_path = os.path.join(output_dir, wav_filename)
                files_to_convert.append((flac_path, wav_path))

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda args: convert_single_file(*args), files_to_convert)

if __name__ == "__main__":
    input_directory = "./data_med"
    output_directory = "./data_wav"
    num_threads = 32  # Adjust the number of threads based on your CPU

    convert_flac_to_wav(input_directory, output_directory, num_workers=num_threads)
