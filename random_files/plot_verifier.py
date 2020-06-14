import librosa
from glob import glob
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy.signal import get_window
from essentia.standard import *

data_dird = '../wav_temp/'
audio_filesd = glob(data_dird + '/*.wav')



# ------------------------------

y,sr=librosa.core.load(audio_filesd[1], sr=44100,duration=10)

# # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# # librosa.display.specshow(D, y_axis='linear')
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Linear-frequency power spectrogram')



S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')

# ----------------------------------




def estimateF0_autoCorr(x_win, fs, minF0, maxF0):
    f0 = 0
    minT0 = int(fs/maxF0)
    
    maxT0 = int(fs/minF0)
    
    maxValAC = -1
    T0 = -1
    for k in range(minT0, maxT0):
        x_win_shifted = np.hstack((np.zeros(k), x_win[:-k]))
        autoCorr = np.dot(x_win, x_win_shifted)
        if autoCorr > maxValAC:
            T0 = k
            maxValAC = autoCorr
    
    f0 = float(fs)/T0
    
    return f0

query = audio_filesd[1]
fs = 44100
minF0 = 50
maxF0 = 200
windowSize = 4096
hopSize = 1024

x_points=[]
y_points=[]


w = get_window('blackman', windowSize)
x = MonoLoader(filename=query, sampleRate=fs)()
startIndexes = np.arange(0, x.size-windowSize, hopSize, dtype=int)
numWindows = startIndexes.size

f0 = np.zeros_like(startIndexes, dtype=float)
f1=np.zeros_like(startIndexes, dtype=float)

for k in range(numWindows):  # framing/windowing
    x_win = x[startIndexes[k]:startIndexes[k] + windowSize]*w  # window applied here
    alpha = estimateF0_autoCorr(x_win, fs, minF0, maxF0)
    if (alpha < maxF0 and alpha > minF0):
        f0[k] = alpha
    f1[k]=(windowSize+(hopSize*k))/fs
    x_points.append(f1[k])
    y_points.append(f0[k])

# ----------------------------------------------


import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from glob import glob
import librosa.display
from scipy.signal import get_window
from essentia.standard import *
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from aubio import source, pitch 
from scipy import signal
from scipy.io import wavfile



# ------------------------------


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



# -------------------------------------------




pitchY = pYAAPT.yaapt(signal, frame_length=40, tda_frame_length=40, f0_min=75, f0_max=600)

# -------------------------------------------------------------


plt.plot(x_points,y_points,'green')
# plt.plot(np.asarray(pitchesYIN), label='YIN', color='red')
# plt.plot(pitchY.samp_values, label='YAAPT', color='blue')
plt.show()