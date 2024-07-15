import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Step 1: Load the GloVe 50d embeddings
def load_glove_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# Step 2: Create the embedding matrix and word-to-index mapping
def create_embedding_matrix(embeddings_dict, embedding_dim):
    word_to_idx = {}
    idx_to_word = []
    embedding_matrix = []

    for i, (word, vector) in enumerate(embeddings_dict.items()):
        word_to_idx[word] = i
        idx_to_word.append(word)
        embedding_matrix.append(vector)

    embedding_matrix = np.array(embedding_matrix)
    return word_to_idx, idx_to_word, embedding_matrix

# Specify the path to the GloVe file and the embedding dimension
glove_file_path = 'glove.6B.100d.txt'  # Replace with your GloVe file path
embedding_dim = 100

# Load the GloVe embeddings
embeddings_dict = load_glove_embeddings(glove_file_path)

# Create the embedding matrix
word_to_idx, idx_to_word, embedding_matrix = create_embedding_matrix(embeddings_dict, embedding_dim)
# Convert the embedding matrix to a PyTorch tensor
embedding_matrix = torch.tensor(embedding_matrix).to('cuda')

# Step 3: Create the embedding layer
embedding_layer = nn.Embedding.from_pretrained(embedding_matrix).to('cuda')

# Function to convert a word to a vector using the embedding layer
def word_to_vector(word):
    idx = word_to_idx.get(word, None)
    # print(word, idx)
    if idx is not None:
        return embedding_layer(torch.tensor([idx]).to('cuda')).squeeze(0)
    else:
        return None


def dot_attention_max(tensor, matrix):
    distance = torch.matmul(tensor, matrix.T)
    norm_tensor = torch.norm(tensor, dim=0)
    norm_matrix = torch.norm(matrix, dim=1)
    similarity = distance / (norm_matrix * norm_tensor)
    # possibility = F.softmax(distance, dim=-1)
    topk_value, topk_idx = torch.topk(similarity, 10)
    return topk_value, topk_idx

def cosine_similarity_max(tensor, matrix):
    # 计算余弦相似度
    tensor = tensor.unsqueeze(0)  # 添加一个维度
    similarities = F.cosine_similarity(tensor, matrix)
    max_value, max_idx = torch.max(similarities, dim=0)
    return similarities, max_value, max_idx

# Example usage
word1 = "man"
vector1 = word_to_vector(word1)
word2 = "woman"
vector2 = word_to_vector(word2)
word3 = 'duke'
vector3 = word_to_vector(word3)
result_vec = -vector1 + vector2 + vector3
# result_cos, _, result_idx_cos = cosine_similarity_max(result_vec, embedding_matrix)
# print(result_cos.max())
result_value_dot, result_idx_dot = dot_attention_max(result_vec, embedding_matrix)

# print(f"Vector for word '{word1}': {vector1}\n '{word2}' : '{vector2}'\n'{word3}' : '{vector3}'")
# print(result_idx_cos.item())
# print(result_vec)
# result_word_cos = idx_to_word[result_idx_cos]
for i in range(10):
    result_word_dot = idx_to_word[result_idx_dot[i]]
    # print('cos:', result_word_cos)
    print(f'{result_word_dot} {(result_value_dot[i].item()*100):.2f}%')
