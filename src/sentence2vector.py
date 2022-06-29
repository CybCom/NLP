import gensim
from gensim.models import Word2Vec, word2vec

import jieba
import numpy as np
from scipy.linalg import norm

from src.path import root_path, output_path


model = Word2Vec.load(root_path + output_path + "word2vec.model")

def vector_similarity(s1, s2):
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(model.vector_size)
        for word in words:
            v += model.wv[word]
        v /= len(words)
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))
