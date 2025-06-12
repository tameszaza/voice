import soundfile as sf

data, samplerate = sf.read('voice/data_wavefake/fake/ljspeech_melgan/LJ001-0001_gen.wav')
print("Sampling rate melgan:", samplerate)
