import streamlit as st
import pickle
import pandas as pd

# Load necessary data
movies_data = pickle.load(open('model/model.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

st.title('Movie Recommendation System')

# Dropdown to select a movie
movies_list = movies_data['title'].values
option = st.selectbox('Select a Movie', movies_list)

# Function to get recommendations
def recommend(movie):
    if movie not in movies_data['title'].values:
        return ["Movie not found in database."]
    movie_index = movies_data[movies_data['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommendations = [movies_data.iloc[i[0]].title for i in movies_list]
    return recommendations

# Button to generate recommendations
if st.button('Recommend'):
    recommendations = recommend(option)
    st.subheader('Recommended Movies:')
    for rec in recommendations:
        st.write(f"- {rec}")
