from CFModel import CFModel
import math
import numpy as np
import pandas as pd
import os

# Reading ratings file
ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1',
                      usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
max_userid = ratings['user_id'].drop_duplicates().max()
max_movieid = ratings['movie_id'].drop_duplicates().max()

# Reading ratings file
users = pd.read_csv('users.csv', sep='\t', encoding='latin-1',
                    usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading ratings file
movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1',
                     usecols=['movie_id', 'title', 'genres'])
K_FACTORS = 50 # The number of dimensional embeddings for movies and users
TEST_USER = 1707 # A random test user (user_id = 2000)


# Use the pre-trained model
trained_model = CFModel(max_userid, max_movieid, K_FACTORS)
# Load weights
trained_model.load_weights('weights.keras')
# Pick a random test user
users[users['user_id'] == TEST_USER]
print(users[users['user_id'] == TEST_USER])

# import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')


# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    return trained_model.rate(user_id - 1, movie_id - 1)


user_ratings = ratings[ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(TEST_USER, x['movie_id']), axis=1)
user_ratings.sort_values(by='rating', 
                         ascending=False).merge(movies, 
                                                on='movie_id', 
                                                how='inner', 
                                                suffixes=['_u', '_m']).head(20)
print(user_ratings.head(20))
# Sort by actual rating and merge with movie details
sorted_user_ratings = (
    user_ratings.sort_values(by='rating', ascending=False)  # Sort by rating in descending order
    .merge(movies, on='movie_id', how='inner', suffixes=['_u', '_m'])  # Merge with movie details
)

# Display the top 20 sorted ratings
print(sorted_user_ratings.head(20))
                         
recommendations = ratings[ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(TEST_USER, x['movie_id']), axis=1)
recommendations.sort_values(by='prediction',
                          ascending=False).merge(movies,
                                                 on='movie_id',
                                                 how='inner',
                                                 suffixes=['_u', '_m']).head(20)
sorted_recommendations = (
    recommendations.sort_values(by='prediction', ascending=False)  # Sort by prediction in descending order
    .merge(movies, on='movie_id', how='inner', suffixes=['_u', '_m'])  # Merge with movie details
)
print(sorted_recommendations.head(20))

print(user_ratings['prediction'].describe())
print(recommendations['prediction'].describe())