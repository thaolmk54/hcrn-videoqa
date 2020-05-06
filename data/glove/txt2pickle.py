import pandas as pd
import pickle
import csv

df = pd.read_csv('glove.840B.300d.txt', sep =" ", quoting=3, header=None, index_col=0)
print("glove file loaded!")
glove = {key: val.values for key, val in df.T.items()}

with open('glove.840.300d.pkl', 'wb') as fp:
    pickle.dump(glove, fp)