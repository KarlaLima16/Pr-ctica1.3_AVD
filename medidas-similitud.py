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
# Calcula la similitud Jaccard entre las películas
def jaccard_similarity(x, y):
    intersection = len(set(x.split()) & set(y.split()))
    union = len(set(x.split()) | set(y.split()))
    return intersection / union

jaccard_sim = np.zeros((len(data), len(data)))

movie_index = 1

# Calcula las puntuaciones de similitud coseno para la película seleccionada
cosine_scores = list(enumerate(cosine_sim[movie_index]))

# Ordena las películas según la similitud coseno en orden descendente
cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)

# Calcula las puntuaciones de similitud Jaccard para la película seleccionada
jaccard_scores = list(enumerate(jaccard_sim[movie_index]))

# Ordena las películas según la similitud Jaccard en orden descendente
jaccard_scores = sorted(jaccard_scores, key=lambda x: x[1], reverse=True)

# Muestra las 5 mejores recomendaciones basadas en similitud coseno
top_coseno_recomendador = cosine_scores[1:6]

# Muestra las 5 mejores recomendaciones basadas en similitud Jaccard
top_jaccard_recomendador = jaccard_scores[1:6]

# muestra las recomendaciones
print("Recomendaciones basadas en similitud coseno:")
for i, score in top_coseno_recomendador:
    print(data.iloc[i]['Title'], data.iloc[i]['Genres'], data.iloc[i]['Actor1'])

print("\nRecomendaciones basadas en similitud Jaccard:")
for i, score in top_jaccard_recomendador:
    print(data.iloc[i]['Title'], data.iloc[i]['Genres'], data.iloc[i]['Actor1'])
