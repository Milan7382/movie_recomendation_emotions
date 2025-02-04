import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

import streamlit as st
from try_new import recommend, get_clean_data  # Import get_clean_data here
import pickle

# Load the pre-trained movie data, vectorizer, and CountVectorizer (cv)
with open('model/movies.pkl', 'rb') as f:
    movies = pickle.load(f)
with open('model/vector.pkl', 'rb') as f:
    vector = pickle.load(f)

# Ensure that cv (CountVectorizer) is also available for use
def load_cv():
    final_movies, vector, cv = get_clean_data()  # This will return cv as well
    return cv

cv = load_cv()  # Now you have cv loaded

# Initialize session state variables if not already present
if 'user_emotion' not in st.session_state:
    st.session_state.user_emotion = ""  # Empty string initially
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'index' not in st.session_state:
    st.session_state.index = 0

# Streamlit App
def app():
    st.title('Movie Recommendation Based on Emotions')

    # **Form to take user input and prevent automatic refresh**
    with st.form(key="emotion_form"):
        user_emotion = st.text_input("How are you feeling today", value=st.session_state.user_emotion)
        submit_button = st.form_submit_button("Get Recommendations")

        if submit_button:
            if user_emotion.strip():  # Ensure valid input
                # **Reset session state before fetching new recommendations**
                st.session_state.recommendations = []  # Clear previous results
                st.session_state.index = 0  # Reset pagination
                st.session_state.user_emotion = user_emotion.strip()  # Store the latest input
                
                # Fetch new recommendations
                st.session_state.recommendations = recommend(st.session_state.user_emotion, movies, vector, cv)
                st.rerun()  # Force UI refresh to apply changes
            else:
                st.warning("Please enter a valid emotion.")

    # **Display recommendations if available**
    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        start_index = st.session_state.index
        end_index = start_index + 5
        displayed_movies = recommendations[start_index:end_index]

        st.write("### Recommended Movies:")
        for movie in displayed_movies:
            st.write(f"- {movie}")

        # **Button to show next 5 movies**
        if end_index < len(recommendations):  # Check if more movies are available
            if st.button('Show Next 5 Movies'):
                st.session_state.index = end_index  # Update index for next batch
                st.rerun()  # Refresh UI with new movies
        else:
            st.write("No more movies to show.")

# Run the app
if __name__ == "__main__":
    app()
