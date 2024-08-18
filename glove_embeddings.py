import numpy as np

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load the GloVe embeddings
glove_file = 'glove.6B/glove.6B.100d.txt'  # Path to the GloVe file
embeddings = load_glove_embeddings(glove_file)
