import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
import copy

path = '../the-movies-dataset/'

md = pd. read_csv(path + 'movies_metadata.csv')
links_small = pd.read_csv(path + 'links_mod.csv')
credits = pd.read_csv(path + 'credits.csv')
keywords = pd.read_csv(path + 'keywords.csv')
# ratings = pd.read_csv(path + 'ratings_small.csv')

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: [str(x).split('-')[0]] if x != np.nan else [])

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')

smd = md[md['id'].isin(links_small)]
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links_small)]

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

# smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres'] + smd['year']
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
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

id_map = pd.read_csv(path + 'links_mod.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
# id_map = id_map.set_index('tmdbId')
indices_map = id_map.set_index('id')
indices_map_for_tmdb = id_map.set_index('movieId')


def hybrid_recommandation(userId, idx, svd, movie_liked_user):
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
    
    try:
        movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        return movies.head(10)
    except:
        return []

def final_res(userId):
    ratings = pd.read_csv(path + 'ratings_mod_2.csv')

    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    data.split(n_folds=10)
    svd = SVD()
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    best_movie = {}
    movie_liked_user = set([int(i) for i in ratings['movieId'][ratings['userId'] == userId]])
    for movie in movie_liked_user:
        recommanded = hybrid_recommandation(userId, movie, svd, movie_liked_user)
        if len(recommanded) == 0:
            continue
        for r in recommanded.values:
            # title = ''.join([i if ord(i) < 128 else '~' for i in r[0]])
            title = r[0]
            if title in best_movie:
                best_movie[title] = max(float(r[2]), best_movie[title])
                continue
            best_movie[title] = float(r[2])

    for idx in movie_liked_user:
        tmdbId = int(indices_map_for_tmdb['id'][idx])
        t = md.loc[md['id'] == tmdbId]['title']
        t = t.values[0]
        if t in best_movie.keys():
            del best_movie[t]

    copy_best_movie = copy.deepcopy(best_movie)
    for key, value in copy_best_movie.items():
        if len(str(key)) > 25:
            del best_movie[key]
    
    best_movie_sorted = sorted(best_movie.items(), key=lambda x: x[1], reverse=True)
    return best_movie_sorted[:7]