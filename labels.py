# Given that patients in pca.py are ordered, labels will be too.
import pandas as pd
import ast
import time
import h5py
import csv
import numpy as np

Y = pd.read_csv("./ptb-xl/ptbxl_database.csv")
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv('./ptb-xl/scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    # Dictionaries are already evaluated.
    # Go through each of the keys of the patient
    for key in y_dic.keys():
        # If a key happens to be a diagnostically significant scp, it will be added to tmp
        if key in agg_df.index:
            # We look through scp statements with the key as index and select its diagnostic class respectively.
            tmp.append(agg_df.loc[key].diagnostic_class)
    if len(tmp) == 0:
        return 404
    # Output array should be: [NORM, MI, STTC, CD, HYP]
    new_arr = [0,0,0,0,0]
    for item in list(set(tmp)):
        if item == "NORM":
            new_arr[0] = 1
        elif item == "MI":
            new_arr[1] = 1  
        elif item == "STTC":
            new_arr[2] = 1
        elif item == "CD":
            new_arr[3] = 1
        elif item == "HYP":
            new_arr[4] = 1
        
    return new_arr

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

labels = Y[Y["diagnostic_superclass"] != 404]
# Iskat bo treba po ecg_id.
labels.to_csv("./csv/labels_included.csv")

file = h5py.File("./pca_data/pca.h5", "r")
data = file["data"]


temp_data = []

for j in labels.ecg_id:
    # These are the ecg_id's essentially subtracted by one t
    # i is the index of the new array()
    index = j - 1
    temp_data.append(data[index])


h5 = h5py.File("./pca_data/processed.h5", "w")
h5.create_dataset('processed', data=temp_data, shape=(21430,50), dtype=np.float32)
h5.close()