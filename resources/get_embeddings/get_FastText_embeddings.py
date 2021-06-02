import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models.fasttext import FastText
from nltk.tokenize.toktok import ToktokTokenizer


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


def get_FastText_embeddings(text_corpus, labels):
    print("getting FastText embeddings")
    train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(
        text_corpus, labels, test_size=0.33, random_state=42
    )
    tokenizer = ToktokTokenizer()
    # tokenize corpus
    tokenized_train = [tokenizer.tokenize(text) for text in train_corpus]
    tokenized_test = [tokenizer.tokenize(text) for text in test_corpus]
    # choose the dimension of embeddings
    ft_num_features = 1000
    # sg decides whether to use the skip-gram model (1) or CBOW (0)
    print("building fasttext model")
    ft_model = FastText(
        tokenized_train,
        vector_size=ft_num_features,
        window=100,
        min_count=2,
        sample=1e-3,
        sg=0,
        epochs=5,
        workers=10,
    )

    # generate document level embeddings
    print("generating document level embeddings")
    avg_ft_train_features = document_vectorizer(
        corpus=tokenized_train, model=ft_model, num_features=ft_num_features
    )
    avg_ft_test_features = document_vectorizer(
        corpus=tokenized_test, model=ft_model, num_features=ft_num_features
    )

    return (
        avg_ft_train_features,
        avg_ft_test_features,
        train_label_names,
        test_label_names,
    )
