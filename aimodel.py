import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('CSV Text Pattern Matching with Machine Learning')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
pattern = st.text_input('Enter a search pattern')

if uploaded_file is not None and pattern != '':
    data = pd.read_csv(uploaded_file)

    # Use TfidfVectorizer to convert text data to a numeric representation
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data.values.ravel())

    # Calculate cosine similarity between search pattern and text data
    pattern_vector = vectorizer.transform([pattern])
    similarities = cosine_similarity(pattern_vector, vectors)

    # Extract rows with highest similarity score
    top_matches = similarities[0].argsort()[-10:][::-1]
    matches = [(data.iloc[i].name, data.iloc[i].values) for i in top_matches]

    # Display matches
    if len(matches) == 0:
        st.write('No matches found')
    else:
        st.write(f'Found {len(matches)} matches:')
        for match in matches:
            st.write(f'Row index: {match[0]}, Values: {match[1]}')
else:
    st.write('Please choose a file and enter a search pattern')
