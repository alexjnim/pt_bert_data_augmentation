import pandas as pd
from sklearn.model_selection import train_test_split
from utils.text_normalizer import normalize_corpus
import logging
from config.logging_config import configure_logger

logger = logging.getLogger(__name__)
logger = configure_logger(logger)


def get_data(reduce_factor: float = None, top_categories: int = None) -> pd.DataFrame:
    original_df = pd.read_json("data/News_Category_Dataset_v2.json", lines=True)
    df = original_df.copy()
    df["text"] = df["headline"] + " " + df["short_description"]
    if top_categories:
        top_cat = df["category"].value_counts().index.tolist()
        df = df[df["category"].isin(top_cat[:5])]

        logger.info(
            f"size of data: {len(original_df)}\nextracting top {top_categories} categories only\nreduce size of data: {len(df)}\n{df['category'].value_counts()}"
        )
    if reduce_factor:
        logger.info(
            f"extracting {reduce_factor} of the data\norginal size of data: {len(df)}"
        )
        _, df = train_test_split(df, test_size=reduce_factor, random_state=42)
        logger.info("reduced size of data: {}".format(len(df)))

    df["text"] = normalize_corpus(
        corpus=df["text"],
        html_stripping=True,
        contraction_expansion=True,
        accented_char_removal=True,
        text_lemmatization=False,
        text_stemming=False,
        special_char_removal=True,
        remove_digits=False,
        stopword_removal=False,
        lower_case=True,
    )

    return df[["category", "text"]]
