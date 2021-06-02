# Sentence-BERT is basically a pre-trained BERT model that is fine-tuned for computing sentence representation. For fine-tuning the pre-trained BERT model, Sentence-BERT uses a Siamese and triplet network architecture, which makes the fine-tuning faster and helps in obtaining accurate sentence embeddings. 


# when we say the pre-trained Sentence-BERT model, it basically implies that we have taken a pre-trained BERT model and fine-tuned it using the Siamese/triplet network architecture. 

from sentence_transformers import models, SentenceTransformer
from sklearn.model_selection import train_test_split
import torch

def get_SentenceTransformer_embeddings(text_corpus, labels):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    # we can use the pre-trained Sentence-BERT model and obtain the fixed-length sentence representation. 
    embedded_features = model.encode(list(text_corpus))
    train_embedded_features, test_embedded_features, train_label_names, test_label_names = train_test_split(
                                                embedded_features,
                                                labels,
                                                test_size=0.33, random_state=42)
    
    return train_embedded_features, test_embedded_features, train_label_names, test_label_names

def get_SentenceTransformer_custom_embeddings(text_corpus, labels):
    # choose your pretrained model
    word_embedding_model = models.Transformer('albert-base-v2')
    word_embedding_model.max_seq_length = 512

    # here we try mean pooling
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print('getting embeddings')
    print(type(text_corpus))
    embedded_features = model.encode(list(text_corpus), convert_to_tensor=True)

    train_embedded_features, test_embedded_features, train_label_names, test_label_names = train_test_split(
                                                embedded_features,
                                                labels,
                                                test_size=0.33, random_state=42)
    
    return train_embedded_features, test_embedded_features, train_label_names, test_label_names