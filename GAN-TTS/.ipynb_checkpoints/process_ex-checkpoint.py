import os
import glob
import pickle
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from utils.audio import convert_audio
import argparse

# Top-level function to resolve pickling issues
def data_prepare(audio_path, mel_path, wav_file):
    mel, audio = convert_audio(wav_file)
    np.save(audio_path, audio, allow_pickle=False)
    np.save(mel_path, mel, allow_pickle=False)
    return audio_path, mel_path, mel.shape[0]

def process_generator_data(wav_dir, output_dir, num_workers=cpu_count(), max_files=None):
    def process(wav_files, output_dir):
        executor = ProcessPoolExecutor(max_workers=num_workers)
        results = []
        for wav_file in wav_files:
            fid = os.path.basename(wav_file).replace('.wav', '.npy')
            audio_path = os.path.join(output_dir, "audio", fid)
            mel_path = os.path.join(output_dir, "mel", fid)
            results.append(executor.submit(partial(data_prepare, audio_path, mel_path, wav_file)))
        return [result.result() for result in tqdm(results)]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mel"), exist_ok=True)

    # Limit number of files if max_files is specified
    wav_files = glob.glob(f'{wav_dir}/**/*.wav', recursive=True)
    if max_files:
        wav_files = wav_files[:max_files]

    metadata = process(wav_files, output_dir)

    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        for meta in metadata:
            f.write('|'.join(map(str, meta)) + '\n')

    print(f"Processed {len(metadata)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process generator .wav files for training.")
    parser.add_argument('--wav_dir', type=str, required=True, help="Path to the directory containing .wav files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for processed data.")
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help="Number of parallel workers for processing.")
    parser.add_argument('--max_files', type=int, default=None, help="Maximum number of files to process.")
    
    args = parser.parse_args()
    process_generator_data(args.wav_dir, args.output_dir, args.num_workers, args.max_files)
