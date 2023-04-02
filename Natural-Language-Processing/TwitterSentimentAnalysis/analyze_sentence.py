import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
import numpy as np
import re


index_to_sentiment = {0: 'negative', 1: 'positive'}
model_path = "model"
tokenizer_path = "tokenizer"
MAX_LEN = 73


def load_tokenizer(tokenizer_path):
    return tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        tokenizer_path)


def load_mnodel(model_path):
    return tf.keras.models.load_model(model_path)


model = load_mnodel(model_path)
tokenizer = load_tokenizer(tokenizer_path)


def clean_tweets(tweet):
    tweet = BeautifulSoup(tweet).get_text()  # remove all html tags
    # replace each word which start from @ (example : @Kenichan) with space
    tweet = re.sub(r'@[A-Za-z0-9]+', ' ', tweet)
    # replace url or links with space
    tweet = re.sub(r'https?://[A-Za-z0-9./]+', ' ', tweet)
    # replace everything exceptspecified in group
    tweet = re.sub(r"[^A-Za-z.!?']", ' ', tweet)
    # replace multple white spaces with single space
    tweet = re.sub(r" +", ' ', tweet)
    return tweet


def predict_sentiment(text):
    text = clean_tweets(text)
    embedding = tokenizer.encode(text)
    embedding = np.expand_dims(embedding, axis=0)
    pad_embedding = tf.keras.preprocessing.sequence.pad_sequences(
        embedding, maxlen=MAX_LEN, value=0, padding='post')
    prediction = model.predict(pad_embedding)[0]
    print(prediction)
    response = {index_to_sentiment[0]: str(round(
        (1-prediction[0])*100, 2))+"%", index_to_sentiment[1]: str(round((prediction[0])*100, 2))+"%"}
    return response
