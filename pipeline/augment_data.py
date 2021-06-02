from tqdm import tqdm
from resources.transformer_augmenter import transformer_augmenter
from nltk.tokenize.toktok import ToktokTokenizer
from config import config


def augment_data(df, verbose=False):

    tokenizer = ToktokTokenizer()
    augmenter = transformer_augmenter()

    n_sentences = round(len(df) * config.n_sentences_percent)

    augmented_sentences = []
    aug_sent_categories = []

    tokenized_text = [tokenizer.tokenize(text) for text in df["text"]]
    for i in tqdm(range(n_sentences)):
        sentence = tokenized_text[i]
        category = df["category"].iloc[i]

        augmented_sentences.extend(
            augmenter.generate(
                sentence,
                new_sent_per_sent=config.new_sent_per_sent,
                num_words_replace=config.num_words_replace,
                list_of_words=False,
                verbose=verbose,
            )
        )

        for _ in range(config.new_sent_per_sent + 1):
            aug_sent_categories.append(category)

    print("number of the original sentences: {}".format(n_sentences))
    print("number of the augmented sentences: {}".format(len(augmented_sentences)))

    return augmented_sentences, aug_sent_categories
