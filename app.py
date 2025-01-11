import streamlit as st
import pandas as pd
import numpy as np
import nltk
import func
import pickle

nltk.download('punkt')       
nltk.download('stopwords')    
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open('model_new.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# App title and description
st.set_page_config(page_title="Spam Detection", page_icon="📧", layout="centered")
st.title("Spam Message Detection")

st.write(
    "Enter a message below to determine if it is spam or not. This app uses a machine learning model to classify your messages."
)

# Input text area
input_message = st.text_area("Enter Your Message Here", height=100, placeholder="Type your message...")

if st.button("Detect"):

    if input_message.strip() == "":
        st.warning("⚠️ Please enter a valid message to analyze.")
    else:

        # Preprocess and predict
        df = func.preprocess(input_message)

        #tokenizing and stop word removal from input
        df['Message'] = df['Message'].apply(func.tokenize_stopWord)

        #tfidf vectorization
        X = vectorizer.transform(df['Message'])
        X = X.toarray()

        #adding more columns
        X = np.hstack((X,np.array(df['Message_length']).reshape(-1,1)))
        X = np.hstack((X,np.array(df['count']).reshape(-1,1)))
        prediction = model.predict(X)

        # Display results in a styled card
        if prediction[0] == 0:  # ham
            st.markdown(
                """
                <div style="padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; background-color: #F6FFED; text-align: center;">
                    <h2 style="color: #4CAF50;">✅ This is Not a Spam Message</h2>
                    <p style="font-size: 16px; color: #4CAF50;">Your message seems safe and legitimate.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:  # Spam
            st.markdown(
                """
                <div style="padding: 20px; border-radius: 10px; border: 2px solid #FF6347; background-color: #FFF2F2; text-align: center;">
                    <h2 style="color: #FF6347;">🚨 This is a Spam Message</h2>
                    <p style="font-size: 16px; color: #FF6347;">Be cautious, this message may be harmful or unsolicited.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Footer
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <p style="text-align: center; font-size: 14px; color: #888888;">
        Built By RunTime Terrors with ❤️  | Spam Detection Model
    </p>
    """,
    unsafe_allow_html=True,
)
