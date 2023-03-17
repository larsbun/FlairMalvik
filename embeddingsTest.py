from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings



# add embeddings to the sentence
word_embeddings = WordEmbeddings('glove')
flair_embeddings = FlairEmbeddings('news-forward')
bert_ml_embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased')

# create a sentence
sentence = Sentence('This is a sentence.')

#word_embeddings.embed(sentence)
#flair_embeddings.embed(sentence)
bert_ml_embeddings.embed(sentence)
print(sentence[0].embedding.shape)

# access the embeddings for the sentence
embeddings = sentence.embedding

# print the shape of the embeddings tensor
print(embeddings.shape)

# print the first 10 values of the embeddings tensor
print(embeddings[:10])