import os
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from Data import Data
from trainingplot import TrainingPlot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

genre_features = Data()
genre_features.load_preprocess_data()

opt = Adam()
batch_size = 20
nb_epochs = 200

filename = 'training_plot.jpg'
plot_losses = TrainingPlot()

X_train, X_test, y_train, y_test = train_test_split(genre_features.data_X, genre_features.data_Y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

mean = X_train.mean(axis=0)
X_train -= mean
X_test -= mean
X_val -= mean
std = X_train.std(axis=0)
X_train /= std
X_test /= std
X_val /= std

print("Training X shape: " + str(X_train.shape))
print("Training Y shape: " + str(y_train.shape))
print("Val X shape: " + str(X_val.shape))
print("Val Y shape: " + str(y_val.shape))
print("Test X shape: " + str(X_test.shape))
print("Test Y shape: " + str(y_test.shape))

input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential()
model.add(LSTM(units=128, dropout=0.4, recurrent_dropout=0.3, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32, dropout=0.4, recurrent_dropout=0.3, return_sequences=False))
model.add(Dense(units=y_train.shape[1], activation='softmax'))

checkpointer = ModelCheckpoint(monitor='val_loss', filepath='weights.hdf5', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=13, verbose=0, mode='auto', baseline=None)

print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=0, callbacks=[checkpointer, es, plot_losses], validation_data=(X_val,y_val))

model.load_weights("weights.hdf5")

print("\nTesting ...")
score, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)
