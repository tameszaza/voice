import soundfile as sf

data, samplerate = sf.read('voice/data_wavefake/real/wavs/LJ001-0001.wav')
print("Sampling rate melgan:", samplerate)
