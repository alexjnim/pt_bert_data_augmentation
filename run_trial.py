import pandas as pd
import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split
from pipeline.get_data import get_data
from pipeline.augment_data import augment_data
from pipeline.vectorize_text import vectorize_text
from pipeline.train_multiple_models import train_multiple_models
from config import config


mlflow.set_experiment("nlp_classification_without_augmentation")

#######################
#       get data
#######################
if Path(config.train_file_path).is_file():
    print("loading augmented data...")
    train_df = pd.read_csv(config.train_file_path)
    test_df = pd.read_csv(config.test_file_path)
else:
    print("training data does not exist\ngenerating data now")
    df = get_data(
        reduce_factor=config.reduce_factor, top_categories=config.top_categories
    )
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
    train_df.to_csv(config.train_file_path, index=False)
    test_df.to_csv(config.test_file_path, index=False)

#######################
#    vectorise text
#######################
# tfidf, word2vec, fasttext, BERT, sentencetransformer
(
    train_corpus,
    test_corpus,
    train_label_names,
    test_label_names,
) = vectorize_text(train_df, test_df, type=config.vectorizer_type)
#######################
#    train model
#######################
scores_df = train_multiple_models(
    train_corpus, test_corpus, train_label_names, test_label_names
)
print(train_df["category"].value_counts(normalize=True))
print(scores_df)
