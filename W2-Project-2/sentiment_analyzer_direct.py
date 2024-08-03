# All this code is present in the notebook file along with the explanations
# This is just a script to run the analyzer GUI directly, without the notebook part (for conveniece)

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
import string

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix

# import tensorflow_hub as hub
# import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Embedding, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk
import joblib
import json

def remove_html_tags(text):
    pattern = re.compile("<.*?>")
    return pattern.sub(r"", text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def remove_punctuations(text):
    punctuation_marks = string.punctuation
    for punctuation in punctuation_marks:
        text = text.replace(punctuation, "")
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_text(text):
    text = remove_html_tags(text)
    text = text.lower()
    text = remove_url(text)
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    return text

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

vectorizer = joblib.load("vectorizer.pkl")

def preprocess_for_lstm(text):
    max_words = 5000  # Maximum number of words to keep based on frequency
    max_length = 100  # Maximum length of sequences (words) or Vocabsize
    text = preprocess_text(text)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=max_length, padding='post', truncating='post')
    return text  #Text will be ready to be put into the LSTM

def preprocess_for_ml(text):
    text = preprocess_text(text)
    text = vectorizer.transform([text])
    return text #Text will be ready to be put into the ML models

#Loading the models
lstm_model = load_model('LSTM_model.keras')
rfc_model = joblib.load("rfc_model.pkl")
lr_model = joblib.load("lr_model.pkl")
mn_model = joblib.load("mn_model.pkl")
bn_model = joblib.load("bn_model.pkl")

# Main constructor
root = tk.Tk()
root.title("Sentiment Analyzer")
root.geometry("800x600")

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rfc_model,
    "MultinomialNB": mn_model,
    "BernoulliNB": bn_model,
    "LSTM": lstm_model
}

# Label for user input
label = tk.Label(root, text="Enter the text to analyze:", font=('Arial', 18))
label.pack(pady=30)

# The text box for user input
TextBox = tk.Text(root, height=3, width=50, font=('Arial', 14))
TextBox.pack()

# Function to handle predictions
def action_predict():
    text = TextBox.get("1.0", tk.END).strip()
    if not text:
        tk.messagebox.showerror("Input Error", "Please enter some text to analyze.")
        return

    model_name = model_var.get()
    model = models[model_name]

    if model_name == "LSTM":
        text = preprocess_for_lstm(text)
        pred = model.predict(text)
        result = "Positive" if pred[0][0] >= 0.5 else "Negative"
        print(result)
    else:
        text = preprocess_for_ml(text)
        pred = model.predict(text)
        result = "Positive" if pred[0] == 1 else "Negative"
        print(result)
    
    result_label.config(text=f"Prediction: {result}")

# Drop-down menu for model selection
model_var = tk.StringVar(value="Logistic Regression")
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=list(models.keys()), state="readonly", font=('Arial', 14))
model_dropdown.pack(pady=20)

# Button to trigger prediction
predict_button = tk.Button(root, text="Predict", font=('Arial', 14), command=action_predict)
predict_button.pack(pady=20)

# Label to display prediction
result_label = tk.Label(root, text="", font=('Arial', 14))
result_label.pack(pady=20)

root.mainloop()