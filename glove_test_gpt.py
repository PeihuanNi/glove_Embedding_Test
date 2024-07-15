import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np

# Load the GloVe model
glove_file = 'glove.6B.50d.txt'
word_vectors = KeyedVectors.load_word2vec_format(glove_file, no_header=True)

# Find the vector for the expression "women - man + king"
result_vector = word_vectors['woman'] - word_vectors['man'] + word_vectors['king']

# Find the most similar words to the resulting vector
most_similar = word_vectors.most_similar([result_vector], topn=10)

# Display the results
print("Most similar words to 'woman - man + king':")
for word, similarity in most_similar:
    print(f"{word}: {similarity}")
