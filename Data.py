import numpy as np
import librosa
import math
import re
import os
from string import digits

class Data:

    hop_length = None
    genre_list = ['blues','hiphop','dubstep','jazz','pop']

    dir_all_files = "./music"
    
    X_preprocessed_data = 'data_input.npy'
    Y_preprocessed_data = 'data_target.npy'
    
    data_X = data_Y = None
    
    def __init__(self):
        self.hop_length = 512
        self.timeseries_length_list = []

    def load_preprocess_data(self):
        self.files_list = self.path_to_audiofiles(self.dir_all_files)

        all_files_list = []
        all_files_list.extend(self.files_list)

        self.data_X, self.data_Y = self.extract_audio_features(self.files_list)
        with open(self.X_preprocessed_data, 'wb') as f:
            np.save(f, self.data_X)
        with open(self.Y_preprocessed_data, 'wb') as f:
            self.data_Y = self.one_hot(self.data_Y)
            np.save(f, self.data_Y)

    def extract_audio_features(self, list_of_audiofiles):
        #timeseries_length = min(self.timeseries_length_list)
        timeseries_length = 128
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 33), dtype=np.float64)
        target = []

        for i, file in enumerate(list_of_audiofiles):
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)

            splits = re.split('[ .]', file)
            genre = re.split('[ /]', splits[1])[2]
            target.append(genre)

            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

            print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))

        return data, np.expand_dims(np.asarray(target), axis=1)

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            remove_digits = str.maketrans('', '', digits)
       	    result = ''.join([i for i in genre_string if not i.isdigit()])
            result = result.translate(remove_digits)
            index = self.genre_list.index(result)
            y_one_hot[i, index] = 1
        return y_one_hot

    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".au"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio
