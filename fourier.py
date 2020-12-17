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

#### MEAN AND STD HERE
mean = 0.006619616772229425
std = 0.012926531971565828

path = ""
sampling_rate = 100

def generate():
    num_patients = 21837

    h5 = h5py.File('./fourier_data/fourier_data.h5', 'w')

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
            data = fourier(data[0])
            # print(data)
            # time.sleep(15)
            # h5['data'][j] = data
            # count += 1
            all_data.append(data)

    h5.create_dataset('data', shape=(num_patients,5988), data=all_data, dtype=np.float32)
    h5.close()


def fourier(data):
    # Fourier analysis
    arr = []
    freqs = fftfreq(1000)
    mask = freqs > 0
    for k in range(12):
        temp = data[:,k]
        fft_vals = fft(temp)
        # To get true theoretical fit
        fft_theo = 2.0*np.abs(fft_vals/1000)
        arr.extend(fft_theo[mask])
    # Shape here : (12*499,1)
    arr = np.array(arr)

    # Let's do the preprocessing at the end
    arr = arr - mean
    arr = arr / std

    return arr

def IncrementalPCA(batch):
    raise NotImplementedError
    
generate()