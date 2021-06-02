reduce_factor = 0.1
top_categories = 5
# all if you want to select all
percent_to_augment = 0.5
new_sent_per_sent = 1
num_words_replace = 3

aug_file_path = (
    "data/augmented_train_data_pta_"
    + str(percent_to_augment)
    + "_rf_"
    + str(reduce_factor)
    + "_nsps_"
    + str(new_sent_per_sent)
    + "_nwr_"
    + str(num_words_replace)
    + ".csv"
)

test_file_path = (
    "data/test_data_rf_"
    + str(reduce_factor)
    + "_nsps_"
    + str(new_sent_per_sent)
    + "_nwr_"
    + str(num_words_replace)
    + ".csv"
)

train_file_path = (
    "data/train_data_rf_"
    + str(reduce_factor)
    + "_nsps_"
    + str(new_sent_per_sent)
    + "_nwr_"
    + str(num_words_replace)
    + ".csv"
)


# tfidf, word2vec, BERT, sentencetransformer
vectorizer_type = "word2vec"
