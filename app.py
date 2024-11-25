from flask import Flask, request, render_template
from CFModel import CFModel
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data and model
ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1',
                      usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
users = pd.read_csv('users.csv', sep='\t', encoding='latin-1',
                    usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1',
                     usecols=['movie_id', 'title', 'genres'])
max_userid = ratings['user_id'].max()
max_movieid = ratings['movie_id'].max()

K_FACTORS = 50
trained_model = CFModel(max_userid, max_movieid, K_FACTORS)
trained_model.load_weights('weights.keras')


# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    return trained_model.rate(user_id - 1, movie_id - 1)


# Route to display the form
@app.route('/')
def index():
    return render_template('index.html')


# Route to process user ID and display recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])  # Get the user ID from the form

    # Ensure user exists
    if user_id not in users['user_id'].values:
        return f"User ID {user_id} does not exist in the database!"

    # Get user ratings
    user_ratings = ratings[ratings['user_id'] == user_id][['user_id', 'movie_id', 'rating']]
    user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(user_id, x['movie_id']), axis=1)

    # Filter recommendations for movies not rated by the user
    recommendations = ratings[~ratings['movie_id'].isin(user_ratings['movie_id'])][['movie_id']].drop_duplicates()
    recommendations['prediction'] = recommendations['movie_id'].apply(lambda movie_id: predict_rating(user_id, movie_id))

    # Sort recommendations by prediction
    sorted_recommendations = (
        recommendations.sort_values(by='prediction', ascending=False)
        .merge(movies, on='movie_id', how='inner', suffixes=['_u', '_m'])
        .head(50)
    )

    return render_template('recommend.html', user_id=user_id, recommendations=sorted_recommendations)


if __name__ == '__main__':
    app.run(debug=True)
