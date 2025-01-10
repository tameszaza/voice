import os
import math
from pydub import AudioSegment

# Input directory containing audio files
input_dir = "test_fake_source"
# Output directory for the trimmed segments
output_dir = "test_fake"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to process audio files
def trim_audio_files():
    supported_formats = [".wav", ".flac", ".mpga"]

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        file_ext = os.path.splitext(filename)[-1].lower()

        if file_ext in supported_formats:
            try:
                # Load the audio file
                audio = AudioSegment.from_file(file_path)
                
                # Get the duration of the audio in seconds
                duration = math.ceil(len(audio) / 1000)
                
                # Generate 6-second segments
                for i in range(0, duration, 6):
                    start_ms = i * 1000
                    end_ms = min((i + 6) * 1000, len(audio))

                    # Extract segment
                    segment = audio[start_ms:end_ms]

                    # Save segment as .flac
                    segment_filename = f"{os.path.splitext(filename)[0]}_segment_{i//6 + 1}.flac"
                    segment_path = os.path.join(output_dir, segment_filename)
                    segment.export(segment_path, format="flac")

                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    trim_audio_files()
