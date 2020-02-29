import pandas as pd
import pickle
import csv

df = pd.read_csv('glove.6B.300d.txt', sep= " ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
print("glove file loaded!")
glove = {key: val.values for key, val in df.T.items()}

with open('glove.6B.300d.pkl', 'wb') as fp:
    pickle.dump(glove, fp)