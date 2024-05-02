import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import csv

import librosa

import numpy as np
import pandas as pd

from keras.src.models.model import model_from_json
from keras import Sequential
from keras.src.layers import Conv1D, BatchNormalization, Activation, Dropout, MaxPooling1D, Flatten, Dense
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class SpeechEmotionModel:
    def __init__(self):
        self.file_list = None

        self.labels = []
        self.features = []

        self.label_encoder = None
        self.encoded_y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None

    def load_data(self, path):
        self.file_list = os.listdir(path)
        for file in self.file_list:
            class_label = self.file_to_emotion(file[6:-16], file[18:-4])
            data = self.extract_features(path + file)

            self.labels.append(class_label)
            self.features.append(data)

    def label_encoding(self):
        y = np.array(self.labels)

        self.label_encoder = LabelEncoder()
        self.encoded_y = to_categorical(self.label_encoder.fit_transform(y))

    def preprocessing(self):
        X = np.array(self.features)
        self.label_encoding()
        X_train, X_test, y_train, y_test = train_test_split(X, self.encoded_y, test_size=0.2, random_state=42)

        self.X_train = np.expand_dims(X_train, axis=2)
        self.X_test = np.expand_dims(X_test, axis=2)

        self.y_train = y_train
        self.y_test = y_test

    def generate_model(self):
        self.model = Sequential()

        self.model.add(Conv1D(256, 8, padding='same', input_shape=(self.X_train.shape[1], 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(128, 8, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(64, 8, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(128, 5, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(128, kernel_regularizer=l2(0.01)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(64, kernel_regularizer=l2(0.01)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.y_train.shape[1]))
        self.model.add(Activation('softmax'))

        opt = Adam(learning_rate=0.0001)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model.summary()

    def start_training(self, batch_size, epochs):
        self.model.fit(self.X_train,
                       self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(self.X_test, self.y_test))

    def save_model(self, name):
        model_json = self.model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(name + '.weights.h5')

        with open('labels', 'w') as f:
            write = csv.writer(f)
            write.writerows(self.labels)

        print("Saved model to disk")

    def load_model(self, model_file_name, weight_file_name=None):
        if not weight_file_name:
            weight_file_name = model_file_name

        json_file = open(model_file_name + '.json', 'r')
        loaded_model_yaml = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_yaml)

        # load weights into new model
        self.model.load_weights(weight_file_name + '.weights.h5')

        with open('labels.csv', newline='') as f:
            reader = csv.reader(f)
            labels = list(reader)

        for label in labels:
            if len(label):
                self.labels.append(''.join(label))

        print("Loaded model from disk")

    def predict_emotions(self, data):
        if not self.label_encoder:
            self.label_encoding()

        pred_data = self.extract_features(data)
        livedf2 = pred_data
        livedf2 = pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T

        twodim = np.expand_dims(livedf2, axis=2)

        livepreds = self.model.predict(twodim, batch_size=32, verbose=1)
        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        livepredictions = (self.label_encoder.inverse_transform(liveabc))

        return livepredictions

    @staticmethod
    def extract_features(data):
        try:
            audio, sample_rate = librosa.load(data, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T, axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", e)
            return None
        return mfccsscaled

    # @staticmethod
    # def file_to_emotion(emotion):
    #     if emotion == '01':
    #         return 'neutral'
    #     elif emotion == '02':
    #         return 'calm'
    #     elif emotion == '03':
    #         return 'happy'
    #     elif emotion == '04':
    #         return 'sad'
    #     elif emotion == '05':
    #         return 'angry'
    #     elif emotion == '06':
    #         return 'fearful'
    #     elif emotion == '07':
    #         return 'disgust'
    #     elif emotion == '08':
    #         return 'surprised'

    @staticmethod
    def file_to_emotion(emotion, gender):
        if emotion == '01' and int(gender) % 2 == 0:
            return 'female_neutral'
        elif emotion == '01' and int(gender) % 2 == 1:
            return 'male_neutral'
        elif emotion == '02' and int(gender) % 2 == 0:
            return 'female_calm'
        elif emotion == '02' and int(gender) % 2 == 1:
            return 'male_calm'
        elif emotion == '03' and int(gender) % 2 == 0:
            return 'female_happy'
        elif emotion == '03' and int(gender) % 2 == 1:
            return 'male_happy'
        elif emotion == '04' and int(gender) % 2 == 0:
            return 'female_sad'
        elif emotion == '04' and int(gender) % 2 == 1:
            return 'male_sad'
        elif emotion == '05' and int(gender) % 2 == 0:
            return 'female_angry'
        elif emotion == '05' and int(gender) % 2 == 1:
            return 'male_angry'
        elif emotion == '06' and int(gender) % 2 == 0:
            return 'female_fearful'
        elif emotion == '06' and int(gender) % 2 == 1:
            return 'male_fearful'
        elif emotion == '07' and int(gender) % 2 == 0:
            return 'female_disgust'
        elif emotion == '07' and int(gender) % 2 == 1:
            return 'male_disgust'
        elif emotion == '08' and int(gender) % 2 == 0:
            return 'female_surprised'
        elif emotion == '08' and int(gender) % 2 == 1:
            return 'male_surprised'
