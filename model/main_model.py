import pandas as pd
from nltk.stem.porter import PorterStemmer

# Initialize Stemmer
ps = PorterStemmer()

# Emotion synonyms dictionary for broader matching with stemming applied
emotions_synonyms = {
    "happy": [ps.stem(word) for word in ["joyful", "content", "pleased", "cheerful", "delighted", "elated", "ecstatic", "excited", "satisfied", "blissful", "grateful", "overjoyed"]],
    "sad": [ps.stem(word) for word in ["unhappy", "sorrowful", "depressed", "downcast", "melancholy", "heartbroken", "gloomy", "mournful", "despondent", "blue", "disheartened", "downhearted"]],
    "angry": [ps.stem(word) for word in ["mad", "anger","furious", "irritated", "enraged", "livid", "annoyed", "fuming", "wrathful", "incensed", "outraged", "agitated", "vexed"]],
    "fear": [ps.stem(word) for word in ["scared", "frightened", "terrified", "anxious", "nervous", "apprehensive", "alarmed", "worried", "horrified", "panicked", "startled", "timid"]],
    "surprise": [ps.stem(word) for word in ["astonished", "amazed", "shocked", "startled", "bewildered", "dumbfounded", "taken aback", "stupefied", "stunned", "dazed", "flabbergasted"]],
    "disgust": [ps.stem(word) for word in ["repulsed", "disdainful", "revulsed", "grossed out", "nauseated", "sickened", "horrified", "appalled", "repelled", "abhorrent", "displeased"]],
    "love": [ps.stem(word) for word in ["affection", "adoration", "devotion", "fondness", "passion", "attachment", "care", "infatuation", "amour", "affinity", "tenderness", "romance"]],
    "lonely": [ps.stem(word) for word in ["isolated", "solitary", "lonesome", "desolate", "forlorn", "abandoned", "alone", "secluded", "detached", "alienated", "withdrawn", "empty"]],
    "excited": [ps.stem(word) for word in ["enthusiastic", "thrilled", "eager", "ecstatic", "elated", "overjoyed", "animated", "pumped", "exhilarated", "overcome", "jubilant", "delighted"]],
    "calm": [ps.stem(word) for word in ["relaxed", "serene", "peaceful", "tranquil", "composed", "collected", "unruffled", "placid", "laid-back", "unperturbed", "quiet", "still"]],
    "confused": [ps.stem(word) for word in ["uncertain", "perplexed", "baffled", "puzzled", "disoriented", "lost", "bewildered", "dazed", "hesitant", "unsure", "flustered", "distraught"]]
}

# Mapping emotions to relevant movie genres with no stemming applied here
data = {
    "emotion": ["happy", "sad", "angry", "fear", "surprise", "disgust", "love", "lonely", "excited", "calm", "confused"],
    "genres": [
        "comedy family",  # Happy
        "drama romance documentary",  # Sad
        "action crime thriller war",  # Angry
        "thriller mystery",  # Fear
        "horror mystery sciencefiction",  # Surprise
        "crime documentary",  # Disgust
        "romance music",  # Love
        "romance drama family",  # Lonely
        "action thriller",  # Excited
        "history drama",  # Calm
        "thriller mystery"  # Confused
    ]
}


# Convert data into a DataFrame
df = pd.DataFrame(data)

# Function to apply stemming to emotions in the DataFrame
def stem_emotions_in_data(df):
    # Stem the emotion column to make it consistent with the input emotion after stemming
    df["emotion"] = df["emotion"].apply(lambda x: ps.stem(x))
    return df

# Apply stemming to the "emotion" column in the data
df = stem_emotions_in_data(df)

# Function to extract the core emotion(s) from user input (apply stemming here)
def extract_emotion(input_emotion):
    # Convert to lowercase and stem the input emotion
    stemmed_input = ps.stem(input_emotion.lower())
    
    print("Stemmed user input:", stemmed_input)  # Debugging output
    
    detected_emotions = []

    # Check for exact emotion match
    for emotion in emotions_synonyms.keys():
        print("Comparing with:", emotion, ps.stem(emotion))  # Debugging output
        if ps.stem(emotion) == stemmed_input:
            detected_emotions.append(emotion)

    # If no exact match, check for synonyms
    if not detected_emotions:
        for emotion, synonyms in emotions_synonyms.items():
            if any(synonym == stemmed_input for synonym in synonyms):
                detected_emotions.append(emotion)

    return detected_emotions  # Return all detected emotions

# Function to apply stemming to genres
def stem_genres(genres):
    return " ".join([ps.stem(genre) for genre in genres.split()])

# Function to get genre prediction based on user emotion
def predict_genre(new_emotion):
    emotions = extract_emotion(new_emotion)  # Extract detected emotions
    
    if not emotions:
        return ["No genres found for the given emotion"]

    predicted_genres = []

    for emotion in emotions:
        match_index = df[df["emotion"] == ps.stem(emotion)].index

        if not match_index.empty:
            predicted_genre = df.iloc[match_index[0]]["genres"]
            predicted_genres.extend(stem_genres(predicted_genre).split())  # Apply stemming to genres

    return list(set(predicted_genres)) if predicted_genres else ["No genres found for the given emotion"]

# Example usage
new_emotion = "loved"
predicted_genres = predict_genre(new_emotion)
print("Predicted genres for '{}':".format(new_emotion), predicted_genres)
