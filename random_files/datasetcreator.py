from notes_extractor import note_diffs
# from sklearn.decomposition import PCA
from glob import glob
# from scipy.spatial import distance
import numpy as np
import pandas as pd
import pickle

# query=note_diffs('Query_wav/',0)
# l_query=len(query)


# pca = PCA(n_components=l_query)

data_dird = '../wav_temp/'
audio_filesd = glob(data_dird + '/*.wav')

# temp1 = note_diffs(data_dird, 0)
# np.asarray(temp1)
# # temp1.reshape(1,-1)
# temp2 =pca.fit_transform(temp1)
# print(l_query,len(temp2))

arr=[]

for i in range (len(audio_filesd)):
    arr.append(np.asarray(note_diffs(data_dird,i)))

# arr= np.vstack(arr)

# print(arr)

# print(type(arr))
# arr=np.asarray(arr)
df=pd.DataFrame(arr)
df.to_pickle('array_hin')
