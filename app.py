import pickle
import streamlit as st
import pandas as pd
from preprocessing_universal import *

with open('model_d1_normal_logreg.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer_d1', 'rb') as f2:
    vect = pickle.load(f2)

st.header("Sentimate - A Machine Learning based depression Detection Tool")
st.write("This tool uses Machine Learning and Natural Language Processing to predict if a person is depressed based on how their feeling. I plan to release a better model which is based on transformers soon.")

input_text = st.text_input("How are you doing today?", "I am very happy today!")
st.subheader("Prediction results: ")
user_df = pd.DataFrame(columns=["text"], data=[[input_text]])
user_df = preprocessing(user_df)
X_user = user_df.tweet
X_user_dtm = vect.transform(X_user.values.astype('U'))
y_user_class = model.predict(X_user_dtm)
for i in y_user_class:
    if i==1:
        st.write("\nResult- Most likely depressed.")
        y_user_prob = model.predict_proba(X_user_dtm)[:, 1]
    elif i==0:
        st.write("\nResult- Most likely not depressed.")
        y_user_prob = model.predict_proba(X_user_dtm)[:, 0]
    else:
        st.write("unexpected error")
        
st.write(f"Probability: {y_user_prob[0]*100:.2f}%")

st.divider()
st.caption("Made by [Achintya Jha](https://achintyajha.com)")
