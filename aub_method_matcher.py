import wave
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import interpolate
import pickle
import argparse

def findSongs():    
    
    def calcDbSongsDistances(dbSongs, str2):
        distances=[]
        for i in range(0,len(dbSongs)):
            distance=dynamicEditDistance(dbSongs[i],str2)
            print("Calculating distance from Audio: ",name[i])
            distances.append(distance)
        return distances
        
    
    def dynamicEditDistance(str1, str2):
        temp= np.zeros((len(str1)+1, len(str2)+1))
        for i in range(0, len(temp[0])):
            temp[0][i] = i
        for i in range(0, len(temp)):
            temp[i][0] = i
        for i in range(1, len(str1)+1):
            for j in range(1, len(str2)+1):
                if str1[i-1]==str2[j-1]:
                    temp[i][j] = temp[i-1][j-1]
                else:
                    temp[i][j] = 1 + min(temp[i-1][j-1], temp[i-1][j], temp[i][j-1])
        return temp[len(str1)][len(str2)]
        
        
    def fetchHummedSong():
        print("\nReference Query Name: ",dff[choice][0],"\n")
        pattern=dff[choice][1]
        return pattern
       
        
    def fetchRankedSongs(res,nm):
        print("\nClosest song and their distance: ")
        while(len(res)>0):
            val=np.amin(res)
            loc=np.where(res==val)
            print(nm[loc][0]," ---> ",val)
            res=np.delete(res,loc)
            nm=np.delete(nm,loc)
    

    with open('./'+args.hum_dataset+'/processed_array_'+args.hum_dataset+'.pkl','rb') as ff:
        dff = np.array(pickle.load(ff))
    
    print("\nHummed Song Index: ")

    for i in range(0,dff.shape[0]):
        print(i,'--->',dff[i][0])

    choice=int(input("\nEnter Index for your hum: "))

    with open('./'+args.tar_dataset+'/processed_array_'+args.tar_dataset+'.pkl','rb') as f:
        df = pickle.load(f)
    df=np.array(df)
    name=df[:,0]
    df=df[:,1]
    hummedSong = fetchHummedSong()
    dbSongs=df
    result=np.array(calcDbSongsDistances(dbSongs,hummedSong))

    t1=np.amin(result)
    t2=np.where(result==t1)

    print("\nClosest Matching Song is: ", name[t2][0],"\n")

    chc=str(input("Want to see detailed results? Type Yes/No "))
    if(chc=='Yes' or chc=='YES' or chc=='yes'):
        fetchRankedSongs(result,name)

parser = argparse.ArgumentParser(description='Query by Humming System Matcher')
parser.add_argument('--tar_dataset', default=None, type=str,required=True, help='write target folder name')
parser.add_argument('--hum_dataset', default=None, type=str,required=True, help='write hummed folder name')
args = parser.parse_args()
    

findSongs()
