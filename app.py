import pickle
import streamlit as st
import pandas as pd
from preprocessing_universal import *

with open('model_d2_normal_logreg.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_d1_normal_logreg.pkl', 'rb') as f:
    model2 = pickle.load(f)


with open('vectorizer_d2', 'rb') as f2:
    vect = pickle.load(f2)

with open('vectorizer_d1', 'rb') as f2:
    vect2 = pickle.load(f2)

st.header("Sentimate - A Machine Learning based depression Detection Tool")
st.write("This tool uses Machine Learning and Natural Language Processing to predict if a person is depressed based on how their feeling. I plan to release a better model which is based on transformers soon.")

input_text = st.text_input("How are you doing today?", "I am very happy today!")
user_df = pd.DataFrame(columns=["text"], data=[[input_text]])
user_df = preprocessing(user_df)
X_user = user_df.tweet
X_user_dtm = vect.transform(X_user.values.astype('U'))
y_user_class = model.predict(X_user_dtm)

X_user2 = user_df.tweet
X_user_dtm2 = vect2.transform(X_user2.values.astype('U'))
y_user_class2 = model2.predict(X_user_dtm2)

if len(y_user_class) == len(y_user_class2):
    for i in range(len(y_user_class)):
        if y_user_class[i] == 1 and y_user_class2[i] == 1:
            res = "Most likely depressed."
            y_user_prob = max(model.predict_proba(X_user_dtm)[
                              :, 1], model2.predict_proba(X_user_dtm2)[:, 1])

        elif y_user_class[i] == 1 and y_user_class2[i] == 0:
            res = "Most likely depressed."
            y_user_prob = model.predict_proba(X_user_dtm)[:, 1]

        elif y_user_class2[i] == 1 and y_user_class[i] == 0:
            res = "Most likely depressed."
            y_user_prob = model2.predict_proba(X_user_dtm2)[:, 1]

        elif y_user_class[i] == 0 and y_user_class2[i] == 0:
            res = "Most likely not depressed."
            y_user_prob = max(model.predict_proba(X_user_dtm)[
                              :, 1], model2.predict_proba(X_user_dtm2)[:, 1])
            
else:
    res = "unexpected error"
    y_user_prob = "unexpected error"

result = res
probability = y_user_prob

st.subheader("Prediction results: ")
st.write(result)
st.divider()
st.caption("Made by [Achintya Jha](https://achintyajha.com)")
