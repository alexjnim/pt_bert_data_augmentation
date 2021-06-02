import mlflow
from numpy import triu_indices
import pandas as pd
from pathlib import Path
from pipeline.get_data import get_data
from pipeline.augment_data import augment_data
from pipeline.vectorize_text import vectorize_text
from pipeline.train_multiple_models import train_multiple_models
from config import config


mlflow.set_experiment("nlp_augmentation_classification")

#######################
#       get data
#######################

if Path(config.aug_file_path).is_file():
    print("loading augmented data...")
    aug_train_df = pd.read_csv(config.aug_file_path)
    test_df = pd.read_csv(config.test_file_path)
else:
    print("augmented data does not exist\ngenerating augmented data now")
    df = get_data(
        reduce_factor=config.reduce_factor, top_categories=config.top_categories
    )
    # augment data
    aug_train_df, test_df = augment_data(df, verbose=True)

#######################
#    vectorise text
#######################
# tfidf, word2vec, fasttext, BERT, sentencetransformer
(
    train_corpus,
    test_corpus,
    train_label_names,
    test_label_names,
) = vectorize_text(aug_train_df, test_df, type=config.vectorizer_type)

#######################
#    train model
#######################
scores_df = train_multiple_models(
    train_corpus, test_corpus, train_label_names, test_label_names, aug_logging=True
)
print(aug_train_df["category"].value_counts(normalize=True))
print(scores_df)
