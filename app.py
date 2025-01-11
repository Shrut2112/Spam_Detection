import streamlit as st
import pandas as pd
import func
import pickle
model=pickle.load(open('D:\Data Science\Major project\Spam detection\model (1).pkl','rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))

input = st.text_area("Enter Your Message Here")
if st.button("Detect Spam or ham"):

    df = func.preprocess(input)
    df['Message'] = df['Message'].apply(func.tokenize_stopWord)

    X = vectorizer.transform(df['Message'])

    predict = model.predict(X)

    if predict[0] == 0:
        st.header("This is not a Spam message")
    else:
        st.header("This is a Spam Message")