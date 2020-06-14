import numpy as np
from glob import glob
import time
import os
from scipy.signal import get_window
from essentia.standard import *
import librosa

def note_diffs(data_dirq, serial_no):

    # data_dirq = 'Query_wav/'
    audio_filesq = glob(data_dirq + '/*.wav')
    # print(len(audio_filesq))

    # data_dird = 'dataset_mid'
    # audio_filesd = glob(data_dird + '/*.mid')
    # print(len(audio_filesd))


    # fs = 44100
    # minF0 = 50
    # maxF0 = 2000
    # windowSize=4096
    # hopSize=1024


    def estimateF0_autoCorr(x_win, fs, minF0, maxF0):
        f0 = 0
        minT0 = int(fs/maxF0)
        #print('min t0',minT0)
        maxT0 = int(fs/minF0)
        #print('max t0',maxT0)
        # Your code starts here
        maxValAC = -1
        T0 = -1
        for k in range(minT0, maxT0):
            x_win_shifted = np.hstack((np.zeros(k), x_win[:-k]))
            autoCorr = np.dot(x_win, x_win_shifted)
            if autoCorr > maxValAC:
                T0 = k
                # print('t0',T0)
                maxValAC = autoCorr
        # print(T0 , maxValAC)
        f0 = float(fs)/T0
        # Your code ends here
        return f0
    # dataa = []
    # datap = []

    # start = time.time()
    # for i in range(1):#len(audio_filesq)):
    # mstr = []

    # pathnameq = os.path.splitext(audio_filesq[0])[0]
    # print(pathnameq)
    # filenameq = os.path.basename(pathnameq)
    # print(filenameq)
    query = audio_filesq[serial_no]
    # qname = os.path.splitext(filenameq)[0]
    # print("hi")
    # print(qname)
    # print("hi")
    fs = 44100
    minF0 = 50
    maxF0 = 200
    windowSize = 4096
    hopSize = 1024
    w = get_window('blackman', windowSize)
    x = MonoLoader(filename=query, sampleRate=fs)()
    startIndexes = np.arange(0, x.size-windowSize, hopSize, dtype=int)
    numWindows = startIndexes.size

    # F0 estimation for each window takes place here
    # initializing the array for f0 values
    f0 = np.zeros_like(startIndexes, dtype=float)
    for k in range(numWindows):  # framing/windowing
        x_win = x[startIndexes[k]:startIndexes[k] + windowSize]*w  # window applied here
        alpha = estimateF0_autoCorr(x_win, fs, minF0, maxF0)
        if (alpha < maxF0 and alpha > minF0):
            f0[k] = alpha
    f1 = []
    f2 = []
    f2.append(0)
    beta = 0
    for i in range(numWindows):
        if(f0[i] != 0):
            f1.append(int(librosa.core.hz_to_midi(f0[i])))
            beta += 1
            if(beta > 1):
                f2.append(f1[beta-1]-f1[beta-2])
    return f2