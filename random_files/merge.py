import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from glob import glob
# import matplotlib.pyplot as plt
import librosa.display
# import numpy as np
from scipy.signal import get_window
from essentia.standard import *
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
# import matplotlib.pyplot as plt
# import numpy as np
from aubio import source, pitch 
from scipy import signal
from scipy.io import wavfile

# sns.set() # Use seaborn's default style to make attractive graphs
# plt.rcParams['figure.dpi'] = 100 # Show nicely large images in this notebook


# snd = parselmouth.Sound("../wav_temp/1268690716832_48601.wav")


# sample_rate, samples = wavfile.read('../wav_temp/1268690716832_48601.wav')
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.gca().invert_yaxis()
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()




data_dird = '../wav_temp/'
audio_filesd = glob(data_dird + '/*.wav')
# ------------------------------
# matplotlib.interactive(True)

# y, sr = librosa.load(librosa.util.example_audio_file(), duration=10)
y,sr=librosa.core.load(audio_filesd[1], sr=44100,duration=10)
print(audio_filesd[0])


D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')


# signal = basic.SignalObj('../')




filename = '../wav_temp/1268690716832_48601.wav'
signal = basic.SignalObj('../wav_temp/1268690716832_48601.wav')

downsample = 1
samplerate = 0
win_s = 1764 // downsample # fft size
hop_s = 441 // downsample # hop size
s = source(filename, samplerate, hop_s)
samplerate = s.samplerate
tolerance = 0.8
pitch_o = pitch("yin", win_s, hop_s, samplerate) 
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)
pitchesYIN = []
confidences = [] 
total_frames = 0
while True:
    samples, read = s()
    pitch = pitch_o(samples)[0]
    pitch = int(round(pitch))
    confidence = pitch_o.get_confidence()
    pitchesYIN += [pitch]
    confidences += [confidence]
    total_frames += read
    if read < hop_s:
        break      


pitchY = pYAAPT.yaapt(signal, frame_length=40, tda_frame_length=40, f0_min=75, f0_max=600)

# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 8))
plt.plot(np.asarray(pitchesYIN), label='YIN', color='green')
plt.plot(pitchY.samp_values, label='YAAPT', color='blue')
plt.show()