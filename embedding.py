import torch
import torch.nn as nn

vocab_size = 10000
embedding_dim = 3

vocab = {'hello': 0, 'world': 1, 'genshin': 2, 'impact': 3, 'start': 4, '!': 5}
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embedding_layer1 = nn.Embedding(vocab_size, embedding_dim)

input_sequence = torch.LongTensor([[0, 1, 2], [3, 4, 5]])

embedding_sequence = embedding_layer(torch.tensor([0, 1, 2]))
embedding_sequence1 = embedding_layer1(torch.tensor(1))

print(embedding_sequence)
print(embedding_sequence1)
for _, layer in embedding_layer.named_modules():
    print(layer.weight.data)
