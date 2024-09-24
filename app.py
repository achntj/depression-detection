import pickle
import pandas as pd
from preprocessing_universal import *

with open('model_d1_normal_logreg.pkl', 'rb') as f:
    model = pickle.load(f)


with open('vectorizer_d1', 'rb') as f2:
    vect = pickle.load(f2)

input_text = input("Enter a tweet here: ")
user_df = pd.DataFrame(columns=["text"], data=[[input_text]])
user_df = preprocessing(user_df)
X_user = user_df.tweet
X_user_dtm = vect.transform(X_user.values.astype('U'))
y_user_class = model.predict(X_user_dtm)
y_user_prob = model.predict_proba(X_user_dtm)[:, 1]
print(y_user_class)
i=0

if y_user_class[i] == 1:
    res = "Most likely depressed."

elif y_user_class[i] == 0:
    res = "Most likely not depressed."


result = res
probability = y_user_prob

print(result)
print(probability)
