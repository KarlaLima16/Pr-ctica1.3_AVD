import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_excel("peliculas.xlsx")

# Llena los valores nulos con una cadena vacía
data.fillna('', inplace=True)

# Combina las columnas de género, director y actores en una sola columna de texto
data['combined_features'] = data['Title'] + ' ' + data['Genres'] + ' ' + data['Actor1']

# Crea una matriz TF-IDF para el texto combinado
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# Calcula la similitud coseno entre las películas
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)