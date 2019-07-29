from settings.data_norm_constants import SEQUENCE_LEN
import numpy as np
import sys, random
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
from keras.models import Sequential


def generator(sentence_list, next_word_list, batch_size, word_indices):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y


def get_model(words, dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=1024))
    model.add(Bidirectional(LSTM(256, return_sequences=False)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model

