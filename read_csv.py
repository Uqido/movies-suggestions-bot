import pandas as pd
import numpy as np
from ast import literal_eval
import csv


path = '../the-movies-dataset/'

def get_md():
    md = pd.read_csv(path + 'movies_metadata.csv', encoding='utf-8')
    md = md.drop([19730, 29503, 35587])
    md['id'] = md['id'].astype('int')
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: [str(x).split('-')[0]] if x != np.nan else [])
    return md


def get_titles():
    md = get_md()
    # titles = set()
    # for title in md['title']:
    #     titles.add(''.join([i if ord(i) < 128 else '~' for i in str(title)]))
    # return titles
    return [str(t) for t in md['title']]

def get_most_poular():
    md = get_md()

    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)

    qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres', 'poster_path']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    # TODO  dovrei ritornare titolo e immagine
    movie = []
    for i in qualified.head(7).values:
        movie.append([str(i[0]), "https://image.tmdb.org/t/p/original/" + str(i[-2])])
    return movie


def add_rating(userId, movie_title, rating):
    md = get_md()

    links_small = pd.read_csv(path + 'links_mod.csv', encoding='utf-8')
    id_map = links_small[['movieId', 'tmdbId']]
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

    smd = md[md['id'].isin(links_small)]
    indices = pd.Series(smd.index, index=smd['title'])

    def convert_int(x):
        try:
            return int(x)
        except:
            return np.nan

    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
    indices_map = id_map.set_index('id')

    with open(path + 'ratings_mod_2.csv', 'a') as csvfile:
        fieldnames = ['useless', 'userId','movieId', 'rating']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        tmdbId = md.loc[md['title'] == movie_title]['id']
        tmdbId = tmdbId.values[0]
        movieId = indices_map['movieId'][tmdbId]

        writer.writerow({'useless':0, 'userId':userId, 'movieId':movieId, 'rating':rating})
