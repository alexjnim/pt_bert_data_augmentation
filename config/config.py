import pathlib

reduce_factor = 0.002
top_categories = 5
test_size = 0.2
# all if you want to select all
percent_to_augment = 0.5
new_sent_per_sent = 1
num_words_replace = 3

# tfidf, word2vec, fastttext, BERT, sentencetransformer
vectorizer_type = "tfidf"

aug_file_name = (
    "augmented_train_data_pta_"
    + str(percent_to_augment)
    + "_rf_"
    + str(reduce_factor)
    + "_nsps_"
    + str(new_sent_per_sent)
    + "_nwr_"
    + str(num_words_replace)
    + "_ts_"
    + str(test_size)
    + ".csv"
)

test_file_name = (
    "test_data_rf_"
    + str(reduce_factor)
    + "_nsps_"
    + str(new_sent_per_sent)
    + "_nwr_"
    + str(num_words_replace)
    + "_ts_"
    + str(test_size)
    + ".csv"
)

train_file_name = (
    "train_data_rf_"
    + str(reduce_factor)
    + "_nsps_"
    + str(new_sent_per_sent)
    + "_nwr_"
    + str(num_words_replace)
    + "_ts_"
    + str(test_size)
    + ".csv"
)

# PROJECT_PATH = pathlib.Path(path_holder.__file__).resolve().parent
PROJECT_PATH = pathlib.Path("").absolute()
DATA_DIR = "data"

aug_file_path = PROJECT_PATH / DATA_DIR / aug_file_name
test_file_path = PROJECT_PATH / DATA_DIR / test_file_name
train_file_path = PROJECT_PATH / DATA_DIR / train_file_name
