import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from main_model import predict_genre
import pickle

ps = PorterStemmer()

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'].lower())
    return L

def get_sentiment(overview):
    return TextBlob(overview).sentiment.polarity

def stem_movie_tags(final_movies):
    final_movies['tags'] = final_movies['tags'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))
    return final_movies

def get_clean_data():
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'popularity']]
    movies.dropna(inplace=True)
    
    movies['genres'] = movies['genres'].apply(convert)
    movies['tags'] = movies['genres']
    movies['sentiment'] = movies['overview'].apply(get_sentiment)
    
    final_movies = movies[['movie_id', 'title', 'tags', 'popularity', 'sentiment']].copy()
    final_movies['tags'] = final_movies['tags'].apply(lambda x: " ".join(x).lower())
    final_movies['popularity'] = np.log1p(final_movies['popularity'])
    
    final_movies = stem_movie_tags(final_movies)
    
    cv = CountVectorizer(max_features=1000, stop_words='english')
    vector = cv.fit_transform(final_movies['tags']).toarray()
    
    return final_movies, vector, cv

def recommend(emotion, final_movies, vector, cv):
    emotion = ps.stem(emotion)  # Apply stemming to user emotion
    predicted_genres = predict_genre(emotion)
    print(f"\nPredicted genres for '{emotion}': {predicted_genres}")
    
    if not predicted_genres:
        return ["No genres found for the given emotion"]
    
    predicted_genres = [ps.stem(genre.lower()) for genre in predicted_genres]
    predicted_vector = cv.transform([" ".join(predicted_genres)])
    genre_similarities = cosine_similarity(predicted_vector, vector)[0]
    
    filtered_movies = final_movies[final_movies['tags'].apply(lambda x: any(genre in x for genre in predicted_genres))]
    
    if filtered_movies.empty:
        return ["No matching movies found!"]
    
    emotion_sentiment = TextBlob(emotion).sentiment.polarity
    
    movie_scores = []
    for i in range(len(filtered_movies)):
        movie = filtered_movies.iloc[i]
        matching_genres = len(set(predicted_genres).intersection(set(movie['tags'].split())))
        similarity_score = genre_similarities[i]
        sentiment_diff = 1 - abs(movie['sentiment'] - emotion_sentiment)
        
        genre_weight_adjustment = 0
        if 'romanc' in predicted_genres:
            genre_weight_adjustment += 0.2
        if 'drama' in predicted_genres:
            genre_weight_adjustment += 0.2
        if 'music' in predicted_genres and 'romanc' not in predicted_genres:
            genre_weight_adjustment -= 0.1
        if 'documentari' in predicted_genres:
            genre_weight_adjustment -= 0.1
        
        score = (similarity_score * 0.3) + (matching_genres * 0.5) + (0.1 * sentiment_diff) + (0.05 * movie['popularity']) + (0.05 * genre_weight_adjustment)
        
        movie_scores.append((movie['title'], score))
    
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [movie[0] for movie in movie_scores[:20]]

def main():
    movies, vector, cv = get_clean_data()
    user_emotion = input("\nEnter your emotion: ").strip().lower()
    user_emotion = ps.stem(user_emotion)  # Apply stemming to user input
    recommendations = recommend(user_emotion, movies, vector, cv)
    
    print("\nRecommended Movies:")
    for rec in recommendations:
        print(rec)
    
    with open('model/movies.pkl', 'wb') as f:
        pickle.dump(movies, f)
    with open('model/vector.pkl', 'wb') as f:
        pickle.dump(vector, f)

if __name__ == '__main__':
    main()
