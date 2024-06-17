from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Citation:
# Reimers, Nils and Gurevych, Iryna (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
# In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.
# URL: http://arxiv.org/abs/1908.10084

class SemanticSearch:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SemanticSearch, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'model'):
            self.model = SentenceTransformer("sentence-transformers/msmarco-distilroberta-base-v2")

    def encode(self, corpus):
        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
        return corpus_embeddings

    def search(self, query, sentences, top_k):
        top_k = min(len(sentences),top_k)
        query_embedding = self.encode(query)
        corpus_embeddings = self.encode(sentences)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results_indices = top_results.indices.cpu().numpy()
        results = np.array(sentences)[results_indices].tolist()
        return results
