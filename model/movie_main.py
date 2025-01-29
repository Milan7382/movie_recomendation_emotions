import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pickle

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count !=3:
            L.append(i['name'])
            count +=1
        else:
            break
    return L

def director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def stem(text):
    ps = PorterStemmer()
    y =[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def get_clean_data():
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview','genres','keywords', 'cast','crew']]
    movies.dropna(inplace=True)
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_3)
    movies['crew'] = movies['crew'].apply(director)

    movies['overview'] = movies['overview'].apply(lambda x:x.split())
    movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
    

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    
    final_movies = movies[['movie_id', 'title','tags']].copy()
    final_movies['tags'] = final_movies['tags'].apply(lambda x: " ".join(x))
    final_movies['tags'] = final_movies['tags'].apply(lambda x: x.lower())

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(final_movies['tags']).toarray()
    final_movies['tags'] = final_movies['tags'].apply(stem)
    similarity = cosine_similarity(vector)
    print(similarity[0])
    
    return final_movies,similarity

def recommend(movie, final_movies,similarity):
    data = get_clean_data()
    movie_index =final_movies[final_movies['title'] == 'Avatar'].index[0]
    print(movie_index)
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x:x[1])[1:6]

    recommendation = []
    for i in movies_list:
        recommendation.append(final_movies.iloc[i[0]].title)
    return recommendation
    



def main():
    final_movies, similarity = get_clean_data()
    movie = "Avatar"  # Example movie
    recommendation = recommend(movie, final_movies, similarity)
    print(f"Recommendations for '{movie}':")
    for rec in recommendation:
        print(rec)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(final_movies,f)
    with open('model/similarity.pkl', 'wb') as f:
        pickle.dump(similarity,f)

    
    

if __name__ == '__main__':
    main()