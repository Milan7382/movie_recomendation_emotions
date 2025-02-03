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

# Streamlit App
def app():
    st.title('Movie Recommendation Based on Emotions')
    
    # Text input for the user's emotion
    user_emotion = st.text_input("Enter your emotion", "happy")  # Default value is "happy"
    
    # Initialize session state to track the index for pagination
    if 'index' not in st.session_state:
        st.session_state.index = 0  # Initialize the starting index
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []  # Store recommendations in session state
    
    # Button to get recommendations
    if st.button('Get Recommendations'):
        if user_emotion:
            # Get the movie recommendations based on the input emotion
            st.session_state.recommendations = recommend(user_emotion, movies, vector, cv)
            st.session_state.index = 0  # Reset index for new recommendations
            st.rerun()  # Force a re-run to update the UI
        else:
            st.write("Please enter a valid emotion.")
    
    # Display the current set of 5 recommended movies
    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        start_index = st.session_state.index
        end_index = start_index + 5
        displayed_movies = recommendations[start_index:end_index]
        
        st.write("Recommended Movies:")
        for movie in displayed_movies:
            st.write(movie)
        
        # Button to show next 5 movies
        if st.button('Show Next 5 Movies'):
            if end_index < len(recommendations):  # Check if more movies are available
                st.session_state.index = end_index  # Update the index for next set of movies
                st.rerun()  # Force a re-run to update the UI
            else:
                st.write("No more movies to show.")

# Run the app
if __name__ == "__main__":
    app()
