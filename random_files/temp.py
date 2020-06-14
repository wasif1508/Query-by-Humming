from notes_extractor import note_diffs
from glob import glob
import pretty_midi
import numpy as np
# from glob import glob
import time
import os
from scipy.signal import get_window
from essentia.standard import *
import librosa


data_dir='../target_files/'
audio_files = glob(data_dir+'/*.mid')
# print(audio_files)
# print(midi_sep(audio_files[3]))
query_file='../query/'
query=glob(query_file+'/*.wav')
quer=query[0]
print(quer)

#-------------------------------------------
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

minF0 = 50
maxF0 = 200
windowSize = 4096
hopSize = 1024
fs=44100

w = get_window('blackman', windowSize)
x = MonoLoader(filename=quer, sampleRate=fs)()
startIndexes = np.arange(0, x.size-windowSize, hopSize, dtype=int)
numWindows = startIndexes.size

f0 = np.zeros_like(startIndexes, dtype=float)
f1=np.zeros_like(startIndexes, dtype=float)

for k in range(numWindows):  # framing/windowing
    x_win = x[startIndexes[k]:startIndexes[k] + windowSize]*w  # window applied here
    alpha = estimateF0_autoCorr(x_win, fs, minF0, maxF0)
    if (alpha < maxF0 and alpha > minF0):
        alpha=librosa.hz_to_midi(alpha)
        f0[k] = int(alpha)
# --------------------------------------
def midi_sep(path):
    
    midi_data = pretty_midi.PrettyMIDI(path)
    note_no = []
    noteon = []
    noteoff = []
   
    for j in range(len(midi_data.instruments)):
        midi_data.instruments[j].remove_invalid_notes()
        if not midi_data.instruments[j].is_drum:
            if midi_data.instruments[j].program in range(0,7):
                for note in midi_data.instruments[j].notes:
                    note_no.append(note.pitch)
                    noteon.append(note.start)
                    noteoff.append(note.end)

    return note_no
# ------------------------------------

def diff_gen (arr):
    l=len(arr)
    arr1=[]
    arr1.append(0)
    for i in range (1,l):
        if(arr[i]!=0):
            arr1.append(int(arr[i]-arr[i-1]))
    return arr1
# -------------------------


def score_crea(arr1,arr2):
    siz=np.abs(len(arr1)-len(arr2))
    max1=10000
    if((len(arr1)-len(arr2))<0):
        temp=arr2
        arr2=arr1
        arr1=temp
    for i in range (0,siz+1):
        # print(arr1[i:(i+len(arr2))])
        # print(arr2[0:])
        dist=0
        for ex in range (len(arr2)):
            dist+=(arr1[i+ex]-arr2[ex])*(arr1[i+ex]-arr2[ex])
        print(dist)
        # print("yo",len(arr1[i:(i+len(arr2))]))
        if (dist < max1):
            max1=dist
        print("yo",max1)
        
    print("hello")
    return max1

# -------------------------
f11=f0[f0 != 0]
ref_t= diff_gen(f11)
ref=np.asarray(ref_t)

# print(f0)
# print("hello")
# print(ref)

max=1000000000
pos=0

for j in range(len(audio_files)):
    ar1=midi_sep(audio_files[j])
    ar2_t=diff_gen(ar1)
    ar2=np.asarray(ar2_t)
    t=score_crea(ar2,ref)
    # print(t)
    # print(ar2[0:(0+len(ar2))]-ref[0:])
    if(max>t):
        max=t
        pos=j

print(audio_files[pos])
# print(arr1[0:(0+len(arr2))]-ref[0:])
# print(audio_files[4])
# print(midi_sep(audio_files[2]))
