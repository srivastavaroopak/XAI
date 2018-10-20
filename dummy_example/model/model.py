# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Model:
    
    def __init__(self, data_dir, model_dir = None):
        self.model_dir = None
        self.data_dir = data_dir
        self.load_data()
        self.build_model()
        self.train_model()
    
    def load_data(self):
        f = open(self.data_dir, 'r')
        self.x = list()
        self.y = list()
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip('\n').split('\t')
            self.x.append([float(line[0]), float(line[1])])
            self.y.append(float(line[2]))
        
        self.x_dim = len(self.x[0])
        self.n_samples = len(self.x)
        self.y_dim = 1
        
        encoder = LabelEncoder()
        encoder.fit(self.y)
        self.encoded_Y = encoder.transform(self.y)
        self.encoded_X = np.array(self.x).astype(float)
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(60, input_dim=self.x_dim, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(self.y_dim, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def train_model(self):
        self.model.fit(self.encoded_X, self.encoded_Y, epochs = 100, verbose = 1)
        
    def predict(self, x):
        x = np.array(x)
        return self.model.predict(x)
        
        
                    
