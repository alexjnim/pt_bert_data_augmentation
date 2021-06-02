import mlflow
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

# df = get_data(reduce_factor=config.reduce_factor, top_categories=config.top_categories)
# #######################
# #    augment data
# #######################
# augmented_sentences, aug_sent_categories = augment_data(df, verbose=False)

# aug_df = pd.DataFrame(
#     list(zip(augmented_sentences, aug_sent_categories)),
#     columns=["text", "category"],
# )


my_file = Path(config.file_path)
if my_file.is_file():
    print("loading augmented data...")
    aug_df = pd.read_csv(config.file_path)
    print(aug_df.head())
else:
    print("augmented data does not exist\ngenerating augmented data now")
    df = get_data(
        reduce_factor=config.reduce_factor, top_categories=config.top_categories
    )
    # augment data
    aug_df = augment_data(df, verbose=False)


#######################
#    vectorise text
#######################
# tfidf, word2vec, fasttext, BERT, sentencetransformer
(
    train_corpus,
    test_corpus,
    train_label_names,
    test_label_names,
) = vectorize_text(aug_df, type=config.vectorizer_type)
#######################
#    train model
#######################
scores_df = train_multiple_models(
    train_corpus, test_corpus, train_label_names, test_label_names
)
print(aug_df["category"].value_counts(normalize=True))
print(scores_df)
