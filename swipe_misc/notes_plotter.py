import os
import matplotlib.pyplot as plt
import glob
import argparse
import numpy as np
import librosa

parser = argparse.ArgumentParser(description='Note Plotting')
parser.add_argument('--dataset', default=None, choices=['hummed', 'target'],type=str,required=True, help='hummed or target')
parser.add_argument('--method', default=None, choices=['yin','yinfft','fcomb','schmitt','mcomb','specacf'],type=str,required=True, help='yin or yinfft or fcomb or schmitt or mcomb or specacf')
args = parser.parse_args()

os.chdir('./'+args.dataset)

for file in glob.glob("*.wav"):
    fi,_=os.path.splitext(file)
    os.system("aubionotes -i" + file+ " -r 44100 -M 0.10 -p "+args.method+" -H 128 > "+"./../img/"+fi+"_not.txt")
    # os.system("aubiopitch -i" + file+ " -r 44100 -p "+args.method+" -H 128 > "+"./../img/"+fi+"_pit.txt")
    f=open('./../img/'+fi+'_not.txt','r')
    # f1=open('./../img/'+fi+'_pit.txt','r')
    dat=f.readlines()
    arr=[]
    for line in dat:
        temp=np.array(line.split()).astype(float)
        if(temp.shape[0]==3):
            arr.append(temp)
    # ll=np.array(f1.read().split()).astype(float)
    # pch = ll[0::2]
    # tm  = ll[1::2]


    figure, axes = plt.subplots(2)

    arr=np.array(arr)
    x=arr[:,0]
    te1=np.where(x>70)
    # print(te1)
    # x=librosa.midi_to_hz(x)
    y1=arr[:,1]
    y2=arr[:,2]
    # print(x)
    x=np.delete(x,te1,axis=0)
    y1=np.delete(y1,te1)
    y2=np.delete(y2,te1)
    y3=(y1+y2)/2
    # print(x)
    # axes[0].set_ylim(45,70)
    # axes[1].set_ylim(45,65)
    axes[1].plot(y3,x,color='r')
    # plt.plot(pch,tm,color='red')
    axes[0].hlines(y=x, xmin=y1, xmax=y2, linewidth=2, color='g')
    figure.tight_layout()
    figure.savefig('./../img/'+fi+'_img.png')
    plt.close()