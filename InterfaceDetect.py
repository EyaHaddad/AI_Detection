import streamlit as st
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger le modèle
model = tf.keras.models.load_model('model.keras')

# Télécharger les stopwords pour le prétraitement du texte
nltk.download('stopwords')

# Fonction de prétraitement et de prédiction
def preprocess_and_predict(text):
    # Prétraitement du texte
    text = text.lower().split()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text if word not in stop_words])

    # Initialiser le Tokenizer pour vectoriser le texte
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts([text])  # Adapter le tokenizer sur le texte

    # Vectoriser le texte
    sequence = tokenizer.texts_to_sequences([text])

    # Padding de la séquence
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='pre', truncating='pre')

    # Prédire avec le modèle
    prediction = model.predict(padded_sequence)

    # Interpréter la prédiction
    if prediction[0][0] > 0.5:
        return "Predicted Label: AI generated"
    else:
        return "Predicted Label: Human generated"

# Interface Streamlit
st.title('Text Prediction with Keras Model')

# Zone de texte pour saisir le texte à prédire
text_to_predict = st.text_area("Enter the text for prediction:")

# Lorsque l'utilisateur clique sur le bouton
if st.button('Predict'):
    if text_to_predict:
        result = preprocess_and_predict(text_to_predict)
        st.write(result)
    else:
        st.write("Please enter some text.")
