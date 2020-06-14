import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import librosa

ss=[]
with open('temp2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        ss+=[row]
ss=np.array(ss)
time=ss[:,0].astype(float)
freq=ss[:,1].astype(float)

plt.plot(time,freq)
plt.show()

output_MIDI = [] 
for j in range(0, len(freq)):
    if(freq[j]!=0):
        d=librosa.hz_to_midi(freq[j])
        d=round(d,0)
        output_MIDI.append(d)

print(output_MIDI)
finalLength = len(output_MIDI)
print(finalLength)
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

print(s)
