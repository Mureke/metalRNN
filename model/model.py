import keras
from settings.model_constants import METLRNN_DROPOUT, METLRNN_USE_DROPOUT
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
import numpy as np
import random
import sys
import io
import os


class MetalRNN(keras.models.Sequential):

    def __init__(self, layers, name):
        super().__init__()
        self.get_model()

    def get_model(self):
        print('Build model...')
        self.add(Embedding(input_dim=len(words), output_dim=1024))
        self.add(Bidirectional(LSTM(256, return_sequences=False)))
        if METLRNN_USE_DROPOUT > 0 and METLRNN_DROPOUT > 0:
            self.add(Dropout(METLRNN_DROPOUT))
        self.add(Dense(len(words)))
        self.add(Activation('softmax'))
