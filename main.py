import pandas as pd
import numpy as np
import wfdb
import ast
import scipy.signal as ss
import matplotlib.pyplot as plt
import time
from scipy.fft import fft, fftfreq, ifft
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import preprocessing
import math
import csv
import h5py
import tensorflow as tf
from tensorflow import keras

class DataGenerator():

    def __init__(self):
        self.fourier_mean = 0.006619616772229425
        self.fourier_std = 0.012926531971565828
        self.sampling_rate = 100
        self.num_patients = 21837
        self.data = None
        self.labels = None
        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv('./ptb-xl/scp_statements.csv', index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

    def fourier(self):
        print("Fourier in progres...")
        all_data = []

        for i in range(22):
            for j in range(1000):
                if i == 0 and j==0:
                    continue
                i_temp = str(i).zfill(2)
                j_temp = str(j).zfill(3)
                filename = f"./ptb-xl/records100/{i_temp}000/{i_temp}{j_temp}_lr"
                try:
                    data = wfdb.rdsamp(filename)
                except:
                    break

                print(filename)
                arr = []

                freqs = fftfreq(1000)
                mask = freqs > 0
                for k in range(12):
                    temp = data[0][:,k]
                    fft_vals = fft(temp)
                    # To get true theoretical fit
                    fft_theo = 2.0*np.abs(fft_vals/1000)
                    arr.extend(fft_theo[mask])
                # Shape here : (12*499,1)
                arr = np.array(arr)

                # Let's do the preprocessing at the end
                arr = arr - self.fourier_mean
                arr = arr / self.fourier_std

                all_data.append(arr)

        self.data = np.array(all_data)
        return self.data

    def pca(self):
        print("PCA in progress...")
        pca = PCA(n_components=50)
        pca.fit(self.data)
        pca_data = pca.transform(self.data)
        self.data = pca_data
        return pca_data

    def aggregate_diagnostic(self, y_dic):
        tmp = []
        # Dictionaries are already evaluated.
        # Go through each of the keys of the patient
        for key in y_dic.keys():
            # If a key happens to be a diagnostically significant scp, it will be added to tmp
            if key in self.agg_df.index:
                # We look through scp statements with the key as index and select its diagnostic class respectively.
                tmp.append(self.agg_df.loc[key].diagnostic_class)
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

        Just check if normal or not

    def create_labels(self):
        print("Labeling in progres...")
        Y = pd.read_csv("./ptb-xl/ptbxl_database.csv")
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(self.aggregate_diagnostic)

        labels = Y[Y["diagnostic_superclass"] != 404]

        temp_data = []

        for j in labels.ecg_id:
            # These are the ecg_id's essentially subtracted by one t
            # i is the index of the new array()
            index = j - 1
            temp_data.append(self.data[index])

        self.data = np.array(temp_data)
        self.labels = np.array(labels["diagnostic_superclass"])


        print("Data shape: ", self.data.shape)
        print("Labels shape: ", self.labels.shape)


        return self.data, self.labels

    def write_to_csv(self, count):
        print("Writing to CSV in progress...")
        with open(f"./pre_model_data/data_{count}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(self.data)
        with open(f"./pre_model_data/labels_{count}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(self.labels)

datagen = DataGenerator()
datagen.fourier()
datagen.pca()
datagen.create_labels()
# Select count to whichever index you want, as demonstrated in write_to_csv function
datagen.write_to_csv(count=1)
