reduce_factor = 0.1
top_categories = 5
# all if you want to select all
n_sentences_percent = 0.5
new_sent_per_sent = 1
num_words_replace = 3

file_path = (
    "data/augmented_data_rf_"
    + str(reduce_factor)
    + "_nsps_"
    + str(new_sent_per_sent)
    + "_nwr_"
    + str(num_words_replace)
    + ".csv"
)


# tfidf, word2vec, BERT, sentencetransformer
vectorizer_type = "fasttext"
