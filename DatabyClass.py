import numpy as np
import librosa
import math
import re
import os
from string import digits

class DatabyClass:

    hop_length = None
    genre_list = ['blues','hiphop','dubstep','jazz','pop']

    dir_blues = "./blues"
    dir_dubstep = "./dubstep"
    dir_hiphop = "./hiphop"
    dir_jazz = "./jazz"
    dir_pop = "./pop"
    
    blues_X_preprocessed_data = 'data_blues.npy'
    blues_Y_preprocessed_data = 'data_blues_y.npy'
    dubstep_X_preprocessed_data = 'data_dubstep.npy'
    dubstep_Y_preprocessed_data = 'data_dubstep_y.npy'
    hiphop_X_preprocessed_data = 'data_hiphop.npy'
    hiphop_Y_preprocessed_data = 'data_hiphop_y.npy'
    jazz_X_preprocessed_data = 'data_jazz.npy'
    jazz_Y_preprocessed_data = 'data_jazz_y.npy'
    pop_X_preprocessed_data = 'data_pop.npy'
    pop_Y_preprocessed_data = 'data_pop_y.npy'

    blues_X = blues_Y = None
    dubstep_X = dubstep_Y = None
    hiphop_X = hiphop_Y = None
    jazz_X = jazz_Y = None
    pop_X = pop_Y = None

    def __init__(self):
        self.hop_length = 512
        self.timeseries_length_list = []

    def load_preprocess_data(self):
        self.blues_list = self.path_to_audiofiles(self.dir_blues)
        self.dubstep_list = self.path_to_audiofiles(self.dir_dubstep)
        self.hiphop_list = self.path_to_audiofiles(self.dir_hiphop)
        self.jazz_list = self.path_to_audiofiles(self.dir_jazz)
        self.pop_list = self.path_to_audiofiles(self.dir_pop)

        all_files_list = []
        all_files_list.extend(self.blues_list)
        all_files_list.extend(self.dubstep_list)
        all_files_list.extend(self.hiphop_list)
        all_files_list.extend(self.jazz_list)
        all_files_list.extend(self.pop_list)

        self.blues_X = self.extract_audio_features(self.blues_list)
        with open(self.blues_X_preprocessed_data, 'wb') as f:
            np.save(f, self.blues_X)
        with open(self.blues_Y_preprocessed_data, 'wb') as f:
            blues = np.zeros((352, 5))
            for i in range(0, 352):
                index = 0
                blues[i, index] = 1
            self.blues_Y = blues
            np.save(f, self.blues_Y)
        
        self.dubstep_X = self.extract_audio_features(self.dubstep_list)
        with open(self.dubstep_X_preprocessed_data, 'wb') as f:
            np.save(f, self.dubstep_X)
        with open(self.dubstep_Y_preprocessed_data, 'wb') as f:
            dubstep = np.zeros((398, 5))
            for i in range(0, 398):
                index = 1
                dubstep[i, index] = 1
            self.dubstep_Y = dubstep
            np.save(f, self.dubstep_Y)

        self.hiphop_X = self.extract_audio_features(self.hiphop_list)
        with open(self.hiphop_X_preprocessed_data, 'wb') as f:
            np.save(f, self.hiphop_X)
        with open(self.hiphop_Y_preprocessed_data, 'wb') as f:
            hiphop = np.zeros((396, 5))
            for i in range(0, 396):
                index = 2
                hiphop[i, index] = 1
            self.hiphop_Y = hiphop
            np.save(f, self.hiphop_Y)
            
        self.jazz_X = self.extract_audio_features(self.jazz_list)
        with open(self.jazz_X_preprocessed_data, 'wb') as f:
            np.save(f, self.jazz_X)
        with open(self.jazz_Y_preprocessed_data, 'wb') as f:
            jazz = np.zeros((386, 5))
            for i in range(0, 386):
                index = 3
                jazz[i, index] = 1
            self.jazz_Y = jazz
            np.save(f, self.jazz_Y)
            
        self.pop_X = self.extract_audio_features(self.pop_list)
        with open(self.pop_X_preprocessed_data, 'wb') as f:
            np.save(f, self.pop_X)
        with open(self.pop_Y_preprocessed_data, 'wb') as f:
            pop = np.zeros((330, 5))
            for i in range(0, 330):
                index = 4
                pop[i, index] = 1
            self.pop_Y = pop
            np.save(f, self.pop_Y)

    def extract_audio_features(self, list_of_audiofiles):
        timeseries_length = 128
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 33), dtype=np.float64)

        for i, file in enumerate(list_of_audiofiles):
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)


            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

        return data


    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".au"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio
