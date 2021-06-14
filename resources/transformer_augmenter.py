# https://www.depends-on-the-definition.com/data-augmentation-with-transformers/

import random
import re
import nltk
from transformers import pipeline
import logging
from config.logging_config import configure_logger

logger = logging.getLogger(__name__)
logger = configure_logger(logger)


class transformer_augmenter:
    """
    Use the pretrained masked language model to generate more
    labeled samples from one labeled sentence.
    """

    def __init__(self):
        self.num_sample_tokens = 10
        self.fill_mask = pipeline(
            "fill-mask", top_k=self.num_sample_tokens, model="distilroberta-base"
        )

    def generate(
        self,
        sentence: str,
        new_sent_per_sent: int = 3,
        num_words_replace: int = 1,
        verbose: bool = False,
        list_of_words: bool = False,
    ) -> list:
        """
        Return a list of n augmented sentences.
        """
        all_sentences = []
        # run as often as tokens should be replaced
        original_sentence = sentence.copy()
        if list_of_words:
            all_sentences.append(sentence)
        else:
            all_sentences.append(" ".join(sentence))
        for _ in range(new_sent_per_sent):
            sentence = original_sentence
            replace_tokens = []
            new_tokens = []
            for __ in range(num_words_replace):
                # join the text
                text = " ".join([word for word in sentence])
                # pick a token not in the POS list below
                replace_pos = "."
                while replace_pos in [
                    ".",
                    "TO",
                    ")",
                    "(",
                    "VBZ",
                    "DT",
                    "FW",
                    "POS",
                    "EX",
                    "''",
                    ":",
                ]:
                    replace_token = random.choice(sentence)
                    replace_pos = nltk.pos_tag([replace_token])[0][1]

                # mask the picked token
                masked_text = text.replace(
                    replace_token, f"{self.fill_mask.tokenizer.mask_token}", 1
                )
                # fill in the masked token with Bert, this will return num_sample_tokens many results, select a random one here
                new_pos = "."
                new_token = "Ġ"
                while new_pos in [".", ")", "(", "''", ":", "POS", "#"] or (
                    new_token == "Ġ"
                    or new_token == "•"
                    or new_token == replace_token
                    or new_token == " "
                ):
                    res = self.fill_mask(masked_text)[
                        random.choice(range(self.num_sample_tokens))
                    ]
                    new_token = res["token_str"]
                    new_token = re.sub(" +", "", new_token)
                    # don't want any errors here
                    try:
                        new_pos = nltk.pos_tag([new_token])[0][1]
                    except:
                        new_pos = "POS"
                # create output samples list
                tmp_sentence, augmented_sentence = sentence.copy(), []

                included = False
                for word in tmp_sentence:
                    if word == replace_token and included == False:
                        augmented_sentence.append((new_token))
                        replace_tokens.append(replace_token)
                        new_tokens.append(new_token)
                        included = True
                    else:
                        augmented_sentence.append(word)
                sentence = augmented_sentence

            # check if augmented sentence already exists
            if augmented_sentence not in all_sentences:
                if list_of_words:
                    all_sentences.append(augmented_sentence)
                else:
                    all_sentences.append(" ".join(augmented_sentence))
                if verbose:
                    print("replace tokens: {}".format(replace_tokens))
                    print(
                        "POS of replace tokens: {}".format(
                            [nltk.pos_tag([token])[0][1] for token in replace_tokens]
                        )
                    )
                    print("new tokens: {}".format(new_tokens))
                    print(
                        "POS of new tokens: {}".format(
                            [nltk.pos_tag([token])[0][1] for token in new_tokens]
                        )
                    )
                    print(
                        "original sentence: {}".format(
                            [word for word in original_sentence]
                        )
                    )
                    print(
                        "augmented sentence: {}\n".format(
                            [word for word in augmented_sentence]
                        )
                    )
            logger.debug(
                f"replace tokens: {replace_tokens}\nPOS of replace tokens: {[nltk.pos_tag([token])[0][1] for token in replace_tokens]}\nnew tokens: {new_tokens}\nPOS of new tokens: {[nltk.pos_tag([token])[0][1] for token in new_tokens]}\noriginal sentence: {[word for word in original_sentence]}\naugmented sentence: {[word for word in augmented_sentence]}"
            )
        return all_sentences
