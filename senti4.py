# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:16:06 2023

@author: Asus
"""
import keras
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix

model = keras.models.load_model("best_model2.hdf5")
sentiment = ['netral', 'negatif', 'positif']

# Function to make predictions
def make_prediction(data):
    max_words = 5000
    max_len = 200
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts([data])  # Mengubah menjadi list data
    
    sequence = tokenizer.texts_to_sequences([data])  # Mengubah teks menjadi urutan token
    test = pad_sequences(sequence, maxlen=max_len)
    prediction = sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]]
    return prediction

# Main program
st.title("Prediksi Sentimen Whatsapp Group")
st.write("Masukkan data percakapan Whatsapp :")

# Create input form
input_form = st.form(key='input_form')
input_data = input_form.text_input(label='Data')

# Perform prediction when submit button is clicked
submit_button = input_form.form_submit_button(label='Prediksi')

if submit_button:
    # Display prediction result
    st.subheader("Hasil Prediksi:")
    
    # Make prediction using model
    prediction = make_prediction(input_data)
    
    # Display the prediction
    st.write(prediction)
