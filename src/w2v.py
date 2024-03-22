from gensim.test.utils import datapath
import gensim.models
import os

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        corpus = self.filepath
        for line in open(corpus):
            # assume there's one document per line, tokens separated by whitespace
            yield self.custom_tokenize(line)

    def custom_tokenize(self, line):
        # Split the line into tokens based on space
        tokens = line.split()
        # Optionally, you can perform additional preprocessing on each token
        tokens = [token.lower() for token in tokens if token.isalnum()]  # Keep only alphanumeric tokens and convert to lowercase
        return tokens

class W2V:
    # custom w2v class for custom dataset
    def __init__(self, filepath, min_count = 1, vector_size = 50, workers = 4, window = 5, sg = 0):
        self.set_params(min_count, vector_size, workers, window, sg)
        self.sentences = MyCorpus(filepath)

    # min_count = The minimum count of words to consider when training the model
    # vector_size = emb dimension
    # window = The maximum distance between a target word and words around the target word
    # sg = The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW
    def train(self):
        self.model = gensim.models.Word2Vec(sentences=self.sentences, min_count=self.min_count, vector_size=self.vector_size,
                                            workers=self.workers, window=self.window, sg=self.sg)
    
    def set_params(self, min_count = 1, vector_size = 5, workers = 4, window = 3, sg = 0):
        self.min_count = min_count
        self.vector_size = vector_size
        self.workers = workers
        self.window = window
        self.sg = sg

    def get_model(self):
        return self.model

    # saves the embeddings with appropriate naming appending with the output_folder
    # mode = src or tgt
    # sample output_folder : '../data/preprocessed/dblp.v12.json.filtered.mt120.ts3'
    # sample output_filename : '../data/preprocessed/dblp.v12.json.filtered.mt120.ts3/src.sg0.d50.w3.txt'
    def save_embeddings(self, output_folder, mode):
        output_filepath = f'{output_folder}/{mode}.sg{self.sg}.d{self.vector_size}.w{self.window}.txt'
        self.model.wv.save_word2vec_format(output_filepath)

        return output_filepath


