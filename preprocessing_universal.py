import re
import pandas as pd
from emot.emo_unicode import UNICODE_EMOJI as UNICODE_EMO # For emojis
from emot.emo_unicode import EMOTICONS_EMO as EMOTICONS # For EMOTICONS
import contractions
import pickle

def text_lower(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('&amp;', 'and')
    return text


def remove_usernames(text):
    user_pattern = re.compile(
        r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z_]+[A-Za-z0-9_]+)')
    return user_pattern.sub(r'USER', text)


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

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'URL', text)


def word_expand(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    return expanded_text


def remove_hashtags(text):
    no_hash = re.compile("#(\w+)")
    return no_hash.sub(r' ', text)


def tokenization(text):
    text = re.split('\W+', text)
    return text


def preprocessing(df):
    df['text_lower'] = df['text'].apply(text_lower)
    df['text_no_user'] = df['text_lower'].apply(remove_usernames)
    df['text_no_emoji'] = df['text_no_user'].apply(convert_emoticons)
    df['text_no_emoji'] = df['text_no_user'].apply(convert_emojis)
    df['text_no_emoji'] = df['text_no_emoji'].apply(
        lambda x: x.replace('_', ' '))
    df['text_no_url'] = df['text_no_emoji'].apply(remove_urls)
    df['text_no_url'] = df['text_no_emoji'].apply(remove_urls)
    df['text_expanded'] = df['text_no_url'].apply(word_expand)
    df['text_no_hash'] = df['text_expanded'].apply(remove_hashtags)
    df['processed_data'] = df['text_no_hash']
    df = df.drop(['text', 'text_lower', 'text_no_user', 'text_no_url', 'text_no_emoji', 'text_expanded',
                 'text_no_hash'], axis=1)
    df.rename(columns={'processed_data': 'tweet'}, inplace=True)

    return df
