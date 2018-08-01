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
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import copy


path = '../the-movies-dataset/'

md = pd. read_csv(path + 'newest_film_metadata.csv')
links = pd.read_csv(path + 'newest_film_links.csv')
del md['useless']
del links['useless']
credits = pd.read_csv(path + 'credits.csv')
keywords = pd.read_csv(path + 'keywords.csv')

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: [str(x).split('-')[0]] if x != np.nan else [])
md['year'] = md['year'].fillna('[]').apply(lambda x: [str(x)] if isinstance(x, int) or isinstance(x, float) or isinstance(x, str) else [])

md['popularity'] = md['popularity'].fillna('[]').apply(lambda x: [str(int(x))] if isinstance(x, float) or isinstance(x, int) else [])

links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

#md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')

smd = md[md['id'].isin(links)]
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links)]

smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

indices = pd.Series(smd.index, index=smd['title'])
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['crew'].apply(get_director)
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x,x])

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]
stemmer = SnowballStemmer('english')
stemmer.stem('dogs')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres'] + smd['year'] + smd['popularity']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
inverse_indices = pd.Series(smd['title'], index=smd.index)

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

id_map = pd.read_csv(path + 'newest_film_links.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
# id_map = id_map.set_index('tmdbId')
indices_map = id_map.set_index('id')
indices_map_for_tmdb = id_map.set_index('movieId')


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

def convert_int_2(x):
    try:
        return int(float(x))
    except:
        return np.nan
ratings['rating'] = ratings['rating'].apply(convert_int_2)

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
model.load_weights('models/embedding_weights_final.h5')



def hybrid_recommandation(userId, idx):
    tmdbId = int(indices_map_for_tmdb['id'][idx])
    title = md.loc[md['id'] == tmdbId]['title']
    title = title.values[0]

    idx = 0
    for i, t in enumerate(inverse_indices.values):
        if t == title:
            idx = i
            break
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title','id']]

    def pred(x):
        # print(np.array([userId]))
        # print(np.array([userId]).shape)
        # print(np.array([indices_map.loc[x]['movieId']]))
        # print(np.array([indices_map.loc[x]['movieId']]).shape)
        return model.predict([np.array([userId]), np.array([indices_map.loc[x]['movieId']])])
    
    # try:
    movies['predict'] = movies['id'].apply(pred)
    movies = movies.sort_values('predict', ascending=False)
    return movies.head(10)
    # except:
    #     return []

userid = "0x5eba"
movie_liked_user = set([int(i) for i in ratings['movieId'][ratings['userId'] == userid]])

idx_movie = {}
for idx, i in enumerate(set(movieid)):
    idx_movie[i] = idx

for movie in movie_liked_user:
    recommanded = hybrid_recommandation(userid, idx_movie[movie])
    print(recommanded)

# print(np.argmax(b_y, 1))

# print(metrics.mean_absolute_error(np.argmax(b_y, 1), 
#     np.argmax(model.predict([b_movieid, b_userid]), 1)))

# print(np.argmax(model.predict([b_movieid, b_userid]), 1))

