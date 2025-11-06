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

# Seed setzten für reprodizierbarkeit
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Encoding von Spalte sentiment + umwandeln in numpy array
label_map = {'negative': 0, 'positive': 1}
y = df['sentiment'].map(label_map).values

# tokenized text wird in string gewandelt und umwandeln numpy array
texts = df['tokens'].apply(lambda toks: ' '.join(toks)).values

# DataSplit, stratify=y == 50% neg & 50% pos
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, y, test_size=0.10, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.10, stratify=y_temp, random_state=SEED
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")


MAX_WORDS = 40_000          # max Anzahl der Wörter, die gelernt werden
MAX_LEN = 200               # sequence length
OOV_TOKEN = "<oov>"         # out-of-vocabulary

# Wörter bekommen int IDs, je häufiger desto niedriger die ID
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train)

# gleich lange seqs
def to_padded(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

X_train_pad = to_padded(X_train)
X_val_pad   = to_padded(X_val)
X_test_pad  = to_padded(X_test)

# Build the LSTM model
EMBED_DIM = 128     #Größe der Wortvektoren (mittlere Datasets 100-300)
LSTM_UNITS = 128    # Neuronen pro Richtung
DROPOUT = 0.3       # 30% schützt vor Overfitting -> wird robuster

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),                         #liest vor- und rückwärts
    GlobalMaxPooling1D(),                                                           #Nimmt die stärksten Aktivierunge
    Dropout(DROPOUT),
    Dense(64, activation='relu'),                                                   #verdichtet Infos zu Mustern
    Dropout(DROPOUT),
    Dense(1, activation='sigmoid')                                                  #prediction
])

#Gewichtsanpassung, bestimmt, was und wie gelernt und bewertet wird
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.build(input_shape=(None, MAX_LEN))
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_auc', mode='max'),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, monitor='val_loss'),
]

# Train
BATCH_SIZE = 128            #128 Sätze aufeinmal, dann Lernschritt
EPOCHS = 12

history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ERGEBNISSE

print("\nValidation performance:")
val_metrics = model.evaluate(X_val_pad, y_val, verbose=0)
for name, val in zip(model.metrics_names, val_metrics):
    print(f"{name}: {val:.4f}")

print("\nTest performance:")
test_metrics = model.evaluate(X_test_pad, y_test, verbose=0)
for name, val in zip(model.metrics_names, test_metrics):
    print(f"{name}: {val:.4f}")

y_pred_proba = model.predict(X_test_pad, batch_size=BATCH_SIZE).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\nClassification report (Test):")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

print("Confusion matrix (Test):")
print(confusion_matrix(y_test, y_pred))