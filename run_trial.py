import pandas as pd
import mlflow
from nltk.tokenize.toktok import ToktokTokenizer
from pipeline.get_data import get_data
from pipeline.augment_data import augment_data
from pipeline.vectorize_text import vectorize_text
from pipeline.train_multiple_models import train_multiple_models
from config import config


mlflow.set_experiment("nlp_classification_without_augmentation")

#######################
#       get data
#######################
df = get_data(
    reduce_factor=config.reduce_factor * 2, top_categories=config.top_categories
)
#######################
#    vectorise text
#######################
# tfidf, word2vec, fasttext, BERT, sentencetransformer
(
    train_corpus,
    test_corpus,
    train_label_names,
    test_label_names,
) = vectorize_text(df, type=config.vectorizer_type)
#######################
#    train model
#######################
scores_df = train_multiple_models(
    train_corpus, test_corpus, train_label_names, test_label_names
)
print(df["category"].value_counts(normalize=True))
print(scores_df)
