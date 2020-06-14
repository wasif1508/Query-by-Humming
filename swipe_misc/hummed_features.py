import pyaudio, wave, sys
import sqlite3 as lite
import math
import sqlite3 as lite
import os.path
from os import listdir, getcwd
from IPython.display import display, Image
import json
from pylab import *
import matplotlib.mlab
import matplotlib.pyplot as plt
from bqplot import pyplot as pl
import wave
import numpy as np
from numpy import matlib
from scipy.io import wavfile
from scipy import signal
from scipy import interpolate
import glob
import csv

def swipep(x, fs, plim, dt, sTHR):
    if not plim:
        plim = [30 , 5000]
    if not dt:
        dt = 0.01
    dlog2p = 1.0/96.0
    dERBs = 0.1
    if not sTHR:
        sTHR = -float('Inf')
    
    t = np.arange(0, len(x)/float(fs), dt )  
    
    dc = 4  
    
    K = 2  
    log2pc = np.arange( np.log2(plim[0]), np.log2(plim[len(plim)-1]) ,dlog2p )
    pc = np.power(2, log2pc)

    S = np.zeros( shape=(len(pc), len(t)) )  
    logWs = np.round_( np.log2( np.multiply(4*K, (np.divide(float(fs), plim) ))))
    ws = np.power(2, np.arange( logWs[1-1], logWs[2-1]-1, -1 ))  
    pO = 4*K * np.divide(fs, ws) 
    d = 1 + log2pc - np.log2( np.multiply(4*K, (np.divide(fs, ws[1-1] ))))
    fERBs = erbs2hz( np.arange( hz2erbs(pc[1-1]/4), hz2erbs(fs/2), dERBs))
    
    for i in range(0, len(ws)):
        dn = round( dc * fs / pO[i] ) 
        will = np.zeros( (int(ws[i]/2), 1) )
        learn = np.reshape(x, -1, order='F')[:, np.newaxis]
        mir = np.zeros( (int(dn + ws[i]/2), 1) )
        xzp = np.vstack((will, learn, mir))
        xk = np.reshape(xzp, len(xzp), order='F')
        w = np.hanning( ws[i] )
        o = max( 0, round( ws[i] - dn ) )
        [ X, f, ti, im] = plt.specgram(xk, NFFT=int(ws[i]), Fs=fs, window=w, noverlap=int(o))
          
        f = np.array(f)
        X1 = np.transpose(X)
        
        ip = interpolate.interp1d( f, X1, kind='linear')(fERBs[:, np.newaxis])
        interpol = ip.transpose(2,0,1).reshape(-1,ip.shape[1])
        interpol1 = np.transpose(interpol)
        M = np.maximum( 0, interpol1 ) 
        L = np.sqrt( M )

        if i==(len(ws)-1):
            j = np.where(d - (i+1) > -1)  
            k = np.where(d[j] - (i+1) < 0) 
        elif i==0:
            j = np.where(d - (i+1) < 1)
            k = np.where(d[j] - (i+1) > 0)
        else:
            j = np.where(abs(d - (i+1)) < 1) 
            k1 = np.arange(0,len(j)) 
            k = np.transpose(k1)
        Si = pitchStrengthAllCandidates( fERBs, L, pc[j] )
        if Si.shape[1] > 1:
            tf=[]
            tf = ti.tolist()
            tf.insert(0, 0)
            del tf[-1]
            ti = np.asarray(tf)
            Si = interpolate.interp1d( ti, Si, 'linear', fill_value=np.nan)(t)
        else:
            Si = matlib.repmat( float('NaN'), len(Si), len(t) )
        k=np.array(k)[0]
        j=np.array(j)[0]
        lambda1 = d[j[k]] - (i+1)
        mu = np.ones( np.size(j) )
        mu[k] = 1 - abs( lambda1 )
        S[j,:] = S[j,:] + np.multiply(((np.kron(np.ones((Si.shape[1], 1)), mu)).transpose()), Si)
        
    p = np.empty((Si.shape[1],))
    p[:] = np.NAN
    s = np.empty((Si.shape[1],))
    s[:] = np.NAN
    for j in range(0, Si.shape[1]):
        s[j] = ( S[:,j] ).max(0)
        i =  np.argmax(S[:,j])
        if s[j] < sTHR: continue
        if i==0:
             p[j]=pc[0]
        elif i==len(pc)-1:
            p[j]=pc[0] 
        else:
            I = np.arange(i-1,i+2)
            tc = np.divide(1, pc[I])
            ntc = ( (tc/tc[1]) - 1 ) * 2*np.pi
            c = np.polyfit( ntc, (S[I,j]), 2 )
            ftc = np.divide(1, np.power(2, np.arange( np.log2(pc[I[0]]), np.log2(pc[I[2]]), 0.0013021 )))
            nftc = ( (ftc/tc[1]) - 1 ) * 2*np.pi
            s[j] = ( np.polyval( c, nftc ) ).max(0)
            k =  np.argmax(np.polyval( c, nftc ) )
            p[j] = 2 ** ( np.log2(pc[I[0]]) + (k-1)/768 )
    p[np.isnan(s)-1] = float('NaN') 
    return p, t, s
        
def pitchStrengthAllCandidates( f, L, pc ):
    hh = np.sum(np.multiply(L, L), axis=0)
    ff = (hh[:, np.newaxis]).transpose()
    sq = np.sqrt( ff )
    
    gh = matlib.repmat( sq, len(L), 1 )
    L = np.divide(L, gh)
    S = np.zeros(( len(pc), len(L[0]) ))
    for j in range (0, (len(pc))-1):
        S[j,:] = pitchStrengthOneCandidate( f, L, pc[j] )
    return S
 
numArr=[]

def is_prime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
def primeArr(n):
    for num in range(1,n+2):    
        if is_prime(num):
            numArr.append(num)
    jg = (np.expand_dims(numArr, axis=1)).transpose()
    return numArr

def pitchStrengthOneCandidate( f, L, pc ):
    n = np.fix( f[-1]/pc - 0.75 )
    k = np.zeros( np.size(f) ) 
    q = f / pc 
    for i in (primeArr(int(n))):

        a = abs( q - i )
        p = a < .25
        k[np.where(p)] = np.cos( 2*math.pi * q[np.where(p)] )
        v = np.logical_and(.25 < a, a < .75)
        pl =  np.cos( 2*np.pi * q[np.where(v)] ) / 2
        k[np.where(v)] = np.cos( 2*np.pi * q[np.where(v)] ) / 2
      
    ff=np.divide(1, f)
   
    k = (k*np.sqrt( ff )) 
    k = k / np.linalg.norm( k[k>0.0] )
    S = np.dot((k[:, np.newaxis]).transpose(), L)
    return S

def hz2erbs(hz):
    erbs = 21.4 * np.log10( 1 + hz/229 )
    return erbs

def erbs2hz(erbs):
    hz = (np.power(10, np.divide(erbs,21.4)) - 1 ) * 229
    return hz

def swipe(WAVE_OUTPUT_FILENAME,file):
    audioPath=WAVE_OUTPUT_FILENAME
    print ("Swipe running",audioPath)
    fs, x = wavfile.read(audioPath)
    np.seterr(divide='ignore', invalid='ignore')
    p,t,s = swipep(x, fs, [100,600], 0.001, 0.3)

    fig = plt.figure()
    plt.plot(p)
    path='/home/wasif/Projects/QbH-2/data2/'
    ttt,ext = os.path.splitext(file)
    fig.savefig(path+ttt+'.png')
    
    print ("Features extracted")
    return findFeatures(p)
    



def findFeatures(p):
    print ("findFeatures running")
    print("Read frequencies of audio and find MIDI notes and pattern")
    freq_est=p
    count = 1
    temp = 1
    output_freq   = [] 
    output_num    = []
    
    output = [[0 for x in range(2)] for x in range(len(freq_est))] 
    print(freq_est)
    for i in range(0, len(freq_est)):
        output_freq.append(freq_est[i])

    
    output_MIDI = [] 
    for j in range(0, len(output_freq)):
        d=69+(12*math.log(float(output_freq[j])/440))/(math.log(2))
        d = round(d,0)
        output_MIDI.append(d)
    finalLength = len(output_freq)
    

    s=""
    for i in range(0,finalLength-1):
        if(output_MIDI[i] < output_MIDI[i+1]):
            if((output_MIDI[i+1]-output_MIDI[i])<=2):
                s += "u"
            else:
                s += "U"
        if(output_MIDI[i] == output_MIDI[i+1]):
            s += "S"
        if(output_MIDI[i] > output_MIDI[i+1]):
            if((output_MIDI[i]-output_MIDI[i+1])<=2):
                s += "d"
            else:
                s += "D"
    
    print ("Pattern:",s)
    return s

    
with open('./data3/processed_file_1.csv', 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        file='Alive_target_1.wav'
        WAVE_OUTPUT_FILENAME = os.path.join(os.getcwd(),file)
        base=os.path.basename(WAVE_OUTPUT_FILENAME)
        afile, ext = os.path.splitext(base)    
        filePath=WAVE_OUTPUT_FILENAME
        sss=swipe(WAVE_OUTPUT_FILENAME,file)
        csv_writer.writerow(['Alive_target_1.wav',sss])
        print ("Processed File: ",afile+ext) 
