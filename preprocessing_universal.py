import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string

def text_lower(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('&amp;','and')
    return text
    
# df['text_lower'] = df['text'].apply(text_lower)

def remove_usernames(text):
    user_pattern = re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z_]+[A-Za-z0-9_]+)')
    return user_pattern.sub(r'USER', text)

# df['text_no_user'] = df['text_lower'].apply(remove_usernames)

from emot.emo_unicode import UNICODE_EMOJI as UNICODE_EMO # For emojis
from emot.emo_unicode import EMOTICONS_EMO as EMOTICONS # For EMOTICONS

# Function for converting emojis into word
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, " ".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    return text

# Function for converting emoticons into word
def convert_emoticons(text):
    for emot in EMOTICONS:
        # Escape emoticon characters for safe regex matching
        text = re.sub(re.escape(emot), " ".join(EMOTICONS[emot].replace(",", "").split()), text)
    
    return text

# df['text_no_emoji'] = df['text_no_user'].apply(convert_emoticons)
# df['text_no_emoji'] = df['text_no_user'].apply(convert_emojis)

# Function for url s
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'URL', text)

#Passing the function to 'text_rare'
# df['text_no_url'] = df['text_no_emoji'].apply(remove_urls)

import contractions

def word_expand(text):
    expanded_words = []       
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    return expanded_text
    
# df['text_expanded'] = df['text_no_url'].apply(word_expand)

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} # Pos tag, used Noun, Verb, Adjective and Adverb
# Function for lemmatization using POS tag
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# df["text_lemma"] = df["text_expanded"].apply(lemmatize_words)


def remove_hashtags(text):
    no_hash = re.compile("#(\w+)")
    return no_hash.sub(r' ', text)

# df['text_no_hash'] = df['text_lemma'].apply(remove_hashtags)

#Creating function for tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text

# df['text_tokenized'] = df['text_no_hash'].apply(lambda x: tokenization(x.lower()))

from nltk.tokenize.treebank import TreebankWordDetokenizer

# df['processed_data'] = [TreebankWordDetokenizer().detokenize(text) for text in df['text_tokenized']]

from textblob import TextBlob
sentiment = []

# for sentence in df['processed_data']: 
#     sentiment.append(TextBlob(sentence).sentiment.polarity*10)
    
def textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity*10
    
# df['sentiment_processed'] = df['processed_data'].apply(textblob_sentiment)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_vader = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    return sentiment_vader.polarity_scores(text)['compound']*10


# df['new_vader_sentiment'] = df['processed_data'].apply(vader_sentiment)

def sent_to_text(sentiment, sentiment2):
    if sentiment < 0 and sentiment2 < 0:
        sentiment = 'neg'
    elif sentiment >= 0 or sentiment2 >= 0:
        sentiment = 'pos'

    return sentiment

# # df['label'] = df[['sentiment_processed', 'new_vader_sentiment']].apply(sent_to_text)
# df['label'] = df.apply(lambda x: sent_to_text(x.sentiment_processed, x.new_vader_sentiment), axis=1)


def preprocessing(df):
    df['text_lower'] = df['text'].apply(text_lower)
    df['text_no_user'] = df['text_lower'].apply(remove_usernames)
    df['text_no_emoji'] = df['text_no_user'].apply(convert_emoticons)
    df['text_no_emoji'] = df['text_no_user'].apply(convert_emojis)
    df['text_no_emoji'] = df['text_no_emoji'].apply(lambda x: x.replace('_', ' '))
    df['text_no_url'] = df['text_no_emoji'].apply(remove_urls)
    df['text_no_url'] = df['text_no_emoji'].apply(remove_urls)
    df['text_expanded'] = df['text_no_url'].apply(word_expand)
    df["text_lemma"] = df["text_expanded"].apply(lemmatize_words)
    df['text_no_hash'] = df['text_lemma'].apply(remove_hashtags)
    df['text_tokenized'] = df['text_no_hash'].apply(lambda x: tokenization(x.lower()))
    df['processed_data'] = [TreebankWordDetokenizer().detokenize(text) for text in df['text_tokenized']]
    df['sentiment_processed'] = df['processed_data'].apply(textblob_sentiment)
    df['new_vader_sentiment'] = df['processed_data'].apply(vader_sentiment)
    df['label'] = df.apply(lambda x: sent_to_text(x.sentiment_processed, x.new_vader_sentiment), axis=1)
    df = df.drop(['text', 'text_lower', 'text_no_user', 'text_no_url', 'text_no_emoji', 'text_expanded', 'text_lemma', 'text_no_hash', 'text_tokenized', 'sentiment_processed', 'new_vader_sentiment'], axis=1)
    df.rename(columns = {'processed_data':'tweet'}, inplace = True)
    
    return df

print("Hello, this function is the updated one. :)")
