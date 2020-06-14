import os
import glob
import csv
import numpy as np
import pickle
import math
import librosa
import argparse

def get_notes(WAVE_OUTPUT_FILENAME,file):
    os.system("aubiopitch -i" + WAVE_OUTPUT_FILENAME+ " -r 44100 -p "+args.method+" -H 128 > "+file+".txt")
    f=open(file+'.txt','r')
    note = np.array(f.read().split()[1::2]).astype(float)
    output_MIDI=[]

    for j in range(0, len(note)):
        if(note[j]!=0):
            d=librosa.hz_to_midi(note[j])
            d=round(d,0)
            output_MIDI.append(d)

    s="" 

    for i in range(0,len(output_MIDI)-1):
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
    return s


parser = argparse.ArgumentParser(description='Query by Humming System')
parser.add_argument('--dataset', default=None, type=str,required=True, help='write folder name')
parser.add_argument('--method', default=None, choices=['yin','yinfft','fcomb','schmitt','mcomb','specacf'],type=str,required=True, help='yin or yinfft or fcomb or schmitt or mcomb or specacf')
args = parser.parse_args()

arr=[]

os.chdir('./'+args.dataset)

for file in glob.glob("*.wav"):
    WAVE_OUTPUT_FILENAME = os.path.join(os.getcwd(),str(file))
    base=os.path.basename(WAVE_OUTPUT_FILENAME)
    afile, ext = os.path.splitext(base)    
    filePath=WAVE_OUTPUT_FILENAME
    file1,_=os.path.splitext(file)
    sss=get_notes(WAVE_OUTPUT_FILENAME,file1)
    arr.append([str(file),sss])
    print ("Processed File: ",afile+ext) 

with open('processed_array_'+args.dataset+'.pkl','wb') as f:
    pickle.dump(arr, f)
