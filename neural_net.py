import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.regularizers import l2
from keras import backend as K

MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300

def build_model(nb_words,word_embedding_matrix):
    
    Q1 = Sequential()
    Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))

    Q2 = Sequential()
    Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))

    model = Sequential()
    model.add(Merge([Q1, Q2], mode='concat'))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model