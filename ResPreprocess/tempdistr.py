from pathlib import Path
import soundfile as sf

def count_duration_bins(wav_dir: str):
    """
    Traverse wav_dir recursively, read each .wav's duration, 
    and count how many fall into each time-interval bin.
    """
    bins = {
        '0-3s':  0,
        '3-6s':  0,
        '6-9s':  0,
        '9s+':   0,
    }

    for wav_path in Path(wav_dir).rglob("*.wav"):
        info = sf.info(wav_path)               # metadata only
        duration = info.frames / info.samplerate

        if duration < 3.0:
            bins['0-3s'] += 1
        elif duration < 6.0:
            bins['3-6s'] += 1
        elif duration < 9.0:
            bins['6-9s'] += 1
        else:
            bins['9s+'] += 1

    return bins

if __name__ == "__main__":
    counts = count_duration_bins("data_train/train/real")
    for interval, n in counts.items():
        print(f"{interval:>5} : {n}")
