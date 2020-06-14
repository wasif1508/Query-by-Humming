import matplotlib.mlab
import wave
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import interpolate
import numpy.matlib
import pandas as pd

def findSongs():
    
    path='./target/'
    
    
    def calcDbSongsDistances(dbSongs, str2):
        distances=[]
        dbSongs1=dbSongs[:,0]
        dbSongs=dbSongs[:,1]
        for i in range(0,len(dbSongs)):
            distance=dynamicEditDistance(dbSongs[i],str2)
            print ("Dist with Audio: ",dbSongs1[i]," is ",distance)
            distances.append(distance)
        return distances
        
    
    def dynamicEditDistance(str1, str2):
        temp= np.zeros((len(str1)+1, len(str2)+1)) #creating a temp of rows of length as of str1 and cols of str2
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
        
    def fetchDBSongs():
        patterns=df
        # print("patterns: ",patterns)
        return patterns
        
    def fetchHummedSong():
        print("Audio Name: ",df[3][0])
        pattern=df[3,1]
        return pattern
       
    def updateDistance():
        #print ("insertInDB ")
        #Insert values in database
        DB = path+'/QBH_MIR.db'
        #print (len(dbSongs))
    
        # connection = lite.connect(DB)
        for i in range(len(dbSongs)):
            with connection:
                cursor = connection.cursor()
                sql = "UPDATE audioFeature SET distance=? WHERE pattern=?"
                connection.execute(sql,[result[i], dbSongs[i]]) 
                connection.commit()
        
    def fetchRankedSongs():
        with connection:
            cursor = connection.cursor() 
            connection.row_factory = lambda cursor, row: row[0]
            c = connection.cursor()
            names = c.execute('SELECT songName FROM audioFeature ORDER BY distance ASC LIMIT 5').fetchall()
            distances = c.execute('SELECT distance FROM audioFeature ORDER BY distance ASC LIMIT 5').fetchall()
            ids = c.execute('SELECT id FROM audioFeature ORDER BY distance ASC LIMIT 5').fetchall()
            songPath = c.execute('SELECT song FROM audioFeature ORDER BY distance ASC LIMIT 5').fetchall()
            imagePath = c.execute('SELECT ImageName FROM audioFeature ORDER BY distance ASC LIMIT 5').fetchall()
            #print ("Dataset distance: ",distance)
            connection.commit()   
        return names,distances,ids,songPath,imagePath
    
    df = pd.read_csv("./target/processed_file_target.csv")
    # print(df.shape)
    df=np.array(df.values.tolist())
    hummedSong = fetchHummedSong()
    dbSongs=fetchDBSongs() # Collection of songs
    result=calcDbSongsDistances(dbSongs,hummedSong) #Creating list of min distances of all songs in DB from hummed Query
    print (result)
    updateDistance()
    songs=[]
    allList=[]
    RankedSongs, distances, ids,songPath,imagePath = fetchRankedSongs()
    for i in range(len(distances)):
        songs=ids[i], RankedSongs[i], distances[i],songPath[i],imagePath[i] 
        # print (songs)
        allList.append(songs)
    print (allList)

    
findSongs()