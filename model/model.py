import keras
from settings.model_constants import DROPOUT, USE_DROPOUT, BATCH_SIZE
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
from keras.models import Sequential
from keras import model
import numpy as np
import random
import sys
import io
import os


def get_model(words):
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=1024))
    model.add(Bidirectional(LSTM(256, return_sequences=False)))
    if USE_DROPOUT > 0 and DROPOUT > 0:
        model.add(Dropout(DROPOUT))
    model.add(Dense(len(words)))
    model.add(Activation('relu'))
