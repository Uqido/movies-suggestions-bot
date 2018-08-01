import numpy as np
import pandas as pd
import os,zipfile,requests
from sklearn import dummy, metrics, cross_validation, ensemble

import keras.backend as K
from keras.layers.merge import Concatenate
from keras.layers import Input, Embedding, Activation, Dense, Dropout, Reshape, BatchNormalization, Flatten
from keras.models import Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse

path = '../the-movies-dataset/'

ratings = pd.read_csv(path + 'newest_film_rating_2.csv')
del ratings['useless']


ratings.movieId = ratings.movieId.astype('category')
ratings.userId = ratings.userId.astype('category')

movieid = ratings.movieId.cat.codes.values
userid = ratings.userId.cat.codes.values

n_movies = max(set(movieid))
n_users = max(set(userid))

embedding_size = 32
nClass = 6

def convert_int(x):
    try:
        return int(float(x))
    except:
        return np.nan
ratings['rating'] = ratings['rating'].apply(convert_int)

y = np.zeros((ratings.shape[0], nClass))
y[np.arange(ratings.shape[0]), ratings['rating']] = 1


movie_input = Input(shape=(1,))
movie_vec = Flatten()(Embedding(n_movies + 1, embedding_size)(movie_input))
movie_vec = Dropout(0.5)(movie_vec)

user_input = Input(shape=(1,))
user_vec = Flatten()(Embedding(n_users + 1, embedding_size)(user_input))
user_vec = Dropout(0.5)(user_vec)

input_vecs = Concatenate()([movie_vec, user_vec])
nn = Dropout(0.5)(Dense(128, activation='relu')(input_vecs))
nn = BatchNormalization()(nn)
nn = Dropout(0.5)(Dense(128, activation='relu')(nn))
nn = BatchNormalization()(nn)
nn = Dense(128, activation='relu')(nn)
result = Dense(nClass, activation='softmax')(nn)

model = Model([movie_input, user_input], result)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# a_movieid, b_movieid, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(movieid, userid, y)

# filepath="models/embedding_weights_{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')
# callbacks_list = [checkpoint]
checkpoint = ModelCheckpoint('models/embedding_weights-{epoch:02d}-{val_acc:.2f}.hdf5' , monitor='val_loss' , verbose=1 , save_best_only=False)

try:
    model.fit([movieid, userid], y, 
            validation_data=([movieid, userid], y),
            batch_size=64, shuffle=True, epochs=20, callbacks=[checkpoint])
except KeyboardInterrupt:
    pass

print(np.argmax(y, 1))

print(metrics.mean_absolute_error(np.argmax(y, 1), 
    np.argmax(model.predict([movieid, userid]), 1)))

print(np.argmax(model.predict([movieid, userid]), 1))

model.save_weights('models/embedding_weights_final.h5')

# model.load_weights('models/embedding_weights_10.h5')
