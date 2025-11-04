import os
import re
import string
import random
import json
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("IMDB Dataset.csv")
#print(df.head())
#print(df.info())

# Duplikate und leere Werte entfernen
df.drop_duplicates(subset='review', inplace=True)
df.dropna(subset=['review', 'sentiment'], inplace=True)

#Zeigt ob Dataset ausgeglichen ist
print(df['sentiment'].value_counts())

def clean(text):
    text = text.lower()                                                 #Lowercase
    text = re.sub(r'\d+', " ", text)                        #Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))    #Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()                #Remove überflüssige Leerzeichen
    return text

# Clean auf jeden Review anwenden
df['cleaned_review'] = df['review'].apply(clean)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# tokenizing, lemmatizing, stopwords
def preprocess(text):
    text = clean(text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# preprocess auf jeden review
df['tokens'] = df['review'].apply(preprocess)
print(df[['review', 'tokens']].head())

