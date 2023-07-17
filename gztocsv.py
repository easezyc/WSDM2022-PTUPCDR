import pandas as pd
import gzip
import json
import tqdm
import random
import os
# # Read the .gz file into a pandas DataFrame
# data1 = pd.read_csv('data/raw/reviews_CDs_and_Vinyl_5.json.gz', compression='gzip')
# data2 = pd.read_csv('data/raw/reviews_Electronics_5.json.gz', compression='gzip')
# data3 = pd.read_csv('data/raw/reviews_Grocery_and_Gourmet_Food_5.json.gz', compression='gzip')
# data4 = pd.read_csv('data/raw/reviews_Movies_and_TV_5.json.gz', compression='gzip')
# data5 = pd.read_csv('data/raw/reviews_Video_Games_5.json.gz', compression='gzip')

# # Write the DataFrame to a .csv file
# data1.to_csv('csv/music.csv', index=False)
# data2.to_csv('csv/electronics.csv', index=False)
# data3.to_csv('csv/food.csv', index=False)
# data4.to_csv('csv/movies.csv', index=False)
# data5.to_csv('csv/games.csv', index=False)

# def gztocsv(file):
#     re = []
#     path = 'data/raw/'+file
#     with gzip.open(path, 'rb') as f:
#         for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
#             line = json.loads(line)
#             re.append([line['reviewerID'], line['asin'], line['overall']])
#     re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
#     # print(self.dealing + ' Mid Done.')
#     re.to_csv('csv/'+file+ '.csv', index=0)

# fileList = ['reviews_CDs_and_Vinyl_5.json.gz', 'reviews_Electronics_5.json.gz', 'reviews_Grocery_and_Gourmet_Food_5.json.gz', 'reviews_Movies_and_TV_5.json.gz', 'reviews_Video_Games_5.json.gz']
# for file in fileList:
#     gztocsv(file)

# re = []
# path = 'data/raw/reviews_Movies_and_TV_5.json.gz'
# with gzip.open(path, 'rb') as f:
#     for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
#         line = json.loads(line)
#         re.append([line['reviewerID'], line['asin'], line['overall']])
# re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
# # print(self.dealing + ' Mid Done.')
# re.to_csv('csv/r1eviews_Video_Games_5.csv', index=0)

movies = pd.read_csv('csv/Movies_and_TV.csv')
music = pd.read_csv('csv/reviews_CDs_and_Vinyl_5.csv')
electronics = pd.read_csv('csv/reviews_Electronics_5.csv')
food = pd.read_csv('csv/reviews_Grocery_and_Gourmet_Food_5.csv')
games = pd.read_csv('csv/reviews_Video_Games_5.csv')

movies_uid = movies['uid']
movies_iid = movies['iid']

music_uid = music['uid']
music_iid = music['iid']

electronics_uid = electronics['uid']
electronics_iid = electronics['iid']

food_uid = food['uid']
food_iid = food['iid']

games_uid = games['uid']
games_iid = games['iid']

movies_electronics_uid = len(set(movies_uid)) + len(set(electronics_uid))
movies_electronics_iid = len(set(movies_iid)) + len(set(electronics_iid))
print("ME_UID: ", movies_electronics_uid)
print("ME_IID: ", movies_electronics_iid)

movies_music_uid = len(set(movies_uid)) + len(set(music_uid))
movies_music_iid = len(set(movies_iid)) + len(set(music_iid))
print("MM_UID: ", movies_music_uid)
print("MM_IID: ", movies_music_iid)

movies_food_uid = len(set(movies_uid)) + len(set(food_uid))
movies_food_iid = len(set(movies_iid)) + len(set(food_iid))
print("MF_UID: ", movies_food_uid)
print("MF_IID: ", movies_food_iid)

electronics_games_uid = len(set(electronics_uid)) + len(set(games_uid))
electronics_games_iid = len(set(electronics_iid)) + len(set(games_iid))
print("EG_UID: ", electronics_games_uid)
print("EG_IID: ", electronics_games_iid)