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

md = pd.read_csv(path + 'final_metadata.csv')
links = pd.read_csv(path + 'final_links.csv')
del md['useless']
del links['useless']
credits = pd.read_csv(path + 'credits.csv')
keywords = pd.read_csv(path + 'keywords.csv')

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: [str(x).split('-')[0]] if x != np.nan else [])
# md['year'] = md['year'].fillna('[]').apply(lambda x: [str(int(x))] if isinstance(x, int) or isinstance(x, float) or isinstance(x, str) else [])
md['year'] = md['year'].fillna('[]').apply(literal_eval)

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

smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres'] + smd['popularity'] + smd['year'] 
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

id_map = pd.read_csv(path + 'final_links.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
# id_map = id_map.set_index('tmdbId')
indices_map = id_map.set_index('id')
indices_map_for_tmdb = id_map.set_index('movieId')

ratings = pd.read_csv(path + 'smaller_final_ratings.csv')
del ratings['useless']
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=10)
svd = SVD()
trainset = data.build_full_trainset()
svd.fit(trainset)
print("done")


def list_movies_seen_user(userId):
    movie_liked_user = set([int(i) for i in ratings['movieId'][ratings['userId'] == userId]])
    titles_movies = []
    for i in movie_liked_user:
        tmdbId = int(indices_map_for_tmdb['id'][idx])
        title = md.loc[md['id'] == tmdbId]['title']
        title = title.values[0]
        titles_movies.append(title)
    return titles_movies



def hybrid_recommandation(userId, idx):
    global svd 

    tmdbId = int(indices_map_for_tmdb['id'][idx])
    title = md.loc[md['id'] == tmdbId]['title']
    title = title.values[0]
    print("Last movie")
    print(title)

    idx = 0
    for i, t in enumerate(inverse_indices.values):
        if t == title:
            idx = i
            break
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:50]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title','id']]

    def pred(x):
        try:
            return svd.predict(userId, indices_map.loc[x]['movieId']).est
        except:
            return 0

    movies['recommanded'] = movies['id'].apply(pred)
    movies = movies.sort_values('recommanded', ascending=False)
    return movies.head(30)


def movies_from_last_one(userId):
    ratings = pd.read_csv(path + 'smaller_final_ratings.csv')
    del ratings['useless']
    
    movie_liked_user = [int(i) for i in ratings['movieId'][ratings['userId'] == userId]]
    last_movie = movie_liked_user[-1]
    if len(movie_liked_user) < 1:
        return []

    recommanded = hybrid_recommandation(userId, last_movie)

    best_movie = {}
    for r in recommanded.values:
        title = r[0]
        if title in best_movie:
            best_movie[title] = max(float(r[2]), best_movie[title])
            continue
        best_movie[title] = float(r[2])

    # for r in recommanded.values:
    #     title = r[0]
    #     if len(title) > 25:
    #         continue
    #     best_movie.add(title)

    # elimino i film che ho gia' visto
    for idx in movie_liked_user:
        tmdbId = int(indices_map_for_tmdb['id'][idx])
        t = md.loc[md['id'] == tmdbId]['title']
        t = t.values[0]
        if t in best_movie:
            del best_movie[t]

    best_movie_sorted = sorted(best_movie.items(), key=lambda x: x[1], reverse=True)
    return best_movie_sorted[:7]


def final_res(userId):
    global svd

    ratings = pd.read_csv(path + 'smaller_final_ratings.csv')
    del ratings['useless']

    movie_liked_user = set([int(i) for i in ratings['movieId'][ratings['userId'] == userId]])
    if len(movie_liked_user) < 1:
        return "", 0

    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    data.split(n_folds=10)
    svd = SVD()
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    
    
    max_rate = 0
    max_moveId = 0
    # movies = smd.iloc[movie_indices][['title','id']]
    # dovrei iterare per tutti i movieId
    for i in md['id'].values:
        try:
            pred = svd.predict(userId, indices_map.loc[i]['movieId']).est
        except:
            continue
        # movieId non deve essere tra quelli che ha gia' fatto il rate
        if pred > max_rate and i not in movie_liked_user:
            max_rate = pred
            max_moveId = i

    # tmdbId = int(indices_map_for_tmdb['id'][max_moveId])
    title = md.loc[md['id'] == max_moveId]['title']
    title = str(title.values[0])

    # potrei ritornare direttamente il titolo
    return title, max_rate
    

    # best_movie = {}
    # for movie in movie_liked_user:
    #     recommanded = hybrid_recommandation(userId, movie)
    #     print(recommanded)
    #     if len(recommanded) == 0:
    #         continue
    #     for r in recommanded.values:
    #         title = r[0]
    #         if title in best_movie:
    #             best_movie[title] = max(float(r[2]), best_movie[title])
    #             continue
    #         best_movie[title] = float(r[2])

    # # elimino i film che ho gia' visto
    # for idx in movie_liked_user:
    #     tmdbId = int(indices_map_for_tmdb['id'][idx])
    #     t = md.loc[md['id'] == tmdbId]['title']
    #     t = t.values[0]
    #     if t in best_movie.keys():
    #         del best_movie[t]

    # copy_best_movie = copy.deepcopy(best_movie)
    # for key, value in copy_best_movie.items():
    #     if len(str(key)) > 25:
    #         del best_movie[key]
    
    # best_movie_sorted = sorted(best_movie.items(), key=lambda x: x[1], reverse=True)
    # return best_movie_sorted[0]