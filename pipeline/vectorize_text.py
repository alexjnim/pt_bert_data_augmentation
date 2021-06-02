import mlflow
from resources.get_embeddings.get_TFIDF_embeddings import get_TFIDF_embeddings
from resources.get_embeddings.get_Word2Vec_embeddings import get_Word2Vec_embeddings
from resources.get_embeddings.get_FastText_embeddings import get_FastText_embeddings
from resources.get_embeddings.get_SentenceTransformer_embeddings import (
    get_SentenceTransformer_embeddings,
)
from resources.get_embeddings.get_BERT_embeddings import get_BERT_embeddings


def vectorize_text(df, type="tfidf"):
    if type == "tfidf":
        return get_TFIDF_embeddings(df["text"], df["category"])
    elif type == "word2vec":
        return get_Word2Vec_embeddings(df["text"], df["category"])
    elif type == "fasttext":
        return get_FastText_embeddings(df["text"], df["category"])
    elif type == "BERT":
        return get_BERT_embeddings(df["text"].to_list(), df["category"].to_list())
