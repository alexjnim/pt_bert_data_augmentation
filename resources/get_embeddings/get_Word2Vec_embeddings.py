from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.tokenize.toktok import ToktokTokenizer
import numpy as np


def document_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)

    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.0
        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.0
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector

    features = [
        average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
        for tokenized_sentence in corpus
    ]
    return np.array(features)


def get_Word2Vec_embeddings(text_corpus, labels, vector_dim=500):
    train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(
        text_corpus, labels, test_size=0.33, random_state=42
    )

    tokenizer = ToktokTokenizer()
    # tokenize corpus
    tokenized_train = [tokenizer.tokenize(text) for text in train_corpus]
    tokenized_test = [tokenizer.tokenize(text) for text in test_corpus]

    # Set values for various parameters
    w2v_num_features = vector_dim  # Word vector dimensionality
    window_context = 30  # Context window size
    min_word_count = 2  # Minimum word count
    sample = 1e-3  # Downsample setting for frequent words
    # build word2vec model
    w2v_model = Word2Vec(
        tokenized_train,
        vector_size=w2v_num_features,
        window=window_context,
        min_count=min_word_count,
        sample=sample,
        sg=1,
        epochs=5,
        workers=10,
    )

    # get document level embeddings
    avg_wv_train_features = document_vectorizer(
        corpus=tokenized_train, model=w2v_model, num_features=w2v_num_features
    )
    avg_wv_test_features = document_vectorizer(
        corpus=tokenized_test, model=w2v_model, num_features=w2v_num_features
    )
    return (
        avg_wv_train_features,
        avg_wv_test_features,
        train_label_names,
        test_label_names,
    )
