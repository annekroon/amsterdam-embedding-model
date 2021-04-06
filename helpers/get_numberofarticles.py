#!/usr/bin/env python3
import json
import os
import glob
import pandas as pd

path_to_jsonfiles = '../trainingset/'

alldicts = []
for file in os.listdir(path_to_jsonfiles):
    full_filename = "{}{}".format(path_to_jsonfiles, file)
    with open(full_filename,'r') as fi:
        mydict = json.load(fi)
        alldicts.append(mydict)

alldicts = [dict(v, id=k) for x in alldicts for k, v in x.items()]

df = pd.DataFrame(alldicts).transpose()
df.columns = df.loc['id']
df_succes = df['succes'].drop('id')
n_per_np = df_succes.sum(axis = 1, skipna = True)
total_n = n_per_np.sum(axis = 0)

print("The total number of articles is: \n\n{} \n\nThe total number of articles aggregated by outlet: \n\n{}". format(total_n, n_per_np))
