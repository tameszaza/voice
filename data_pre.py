import os
from pydub import AudioSegment

def convert_flac_to_wav(input_dir, output_dir):
    """
    Converts all .flac files in a given directory and its subdirectories to .wav files,
    saving them in a single output directory.

    :param input_dir: Root directory containing .flac files.
    :param output_dir: Directory to save the converted .wav files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                flac_path = os.path.join(root, file)
                wav_filename = os.path.splitext(file)[0] + ".wav"
                wav_path = os.path.join(output_dir, wav_filename)

                # Convert .flac to .wav
                try:
                    audio = AudioSegment.from_file(flac_path, format="flac")
                    audio.export(wav_path, format="wav")
                    print(f"Converted: {flac_path} -> {wav_path}")
                except Exception as e:
                    print(f"Failed to convert {flac_path}: {e}")

if __name__ == "__main__":
    input_directory = "./data_small"
    output_directory = "./data_wav"
    convert_flac_to_wav(input_directory, output_directory)
