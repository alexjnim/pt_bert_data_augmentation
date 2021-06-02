""" read here for more information:
https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/

notice here that the hidden states for BERT is 512, meaning the text inputs are limited to 510 words (+ [CLS] and [SEP] tokens)
but what if we are working with text longer than 512? Like a news article?
Try chunking and averagine the output vectors:
https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f
"""
import transformers
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def get_BERT_embeddings(text_corpus, labels):
    """return BERT embeddings for classification

    Args:
        text_corpus (list or pandas Series): contains all the normalized text entries
        labels (list): the training targets

    Returns:
        train_embedded_features, test_embedded_features, train_label_names, test_label_names
    """

    # Here we have imported DistilBertModel class and its tokenizer from transformer library, however, feel free to use any model class you see fit
    model_class, tokenizer_class, pretrained_weights = (
        DistilBertModel,
        DistilBertTokenizer,
        "distilbert-base-uncased",
    )
    # Want BERT instead of distilBERT? Uncomment the following line:
    # model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    if max([len(x) for x in text_corpus]) > 510:
        print("length of longest text: {}".format(max([len(x) for x in text_corpus])))
        print("chunking required")
        embedded_features = get_embeddings_for_more_than_512(
            tokenizer, model, text_corpus, use_mean_pooling=True
        )
    else:
        print("no chunking required")
    embedded_features = get_embeddings_for_less_than_512(
        tokenizer, model, text_corpus, use_mean_pooling=True
    )

    # here we don't need to get embeddings for train and test separately as we don't have to 'fit' the DIstilBERT embedding model
    (
        train_embedded_features,
        test_embedded_features,
        train_label_names,
        test_label_names,
    ) = train_test_split(embedded_features, labels, test_size=0.33, random_state=42)

    return (
        train_embedded_features,
        test_embedded_features,
        train_label_names,
        test_label_names,
    )


def mean_pooling(model_output, attention_mask):
    """this should generate embeddings from the model_output and the attention_mask by taking the mean of all the output tokens

    Args:
        model_output (tensor): contains all the out model_output tokens
        attention_mask (tensor): contains all the attention_masks used for the model input

    Returns:
        embeddings: the mean pooled embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings_for_less_than_512(
    tokenizer, model, text_corpus, use_mean_pooling=True
):
    """
    get the input_ids and attention_masks
    need to truncate in case text is longer than 512 words as this BERT model is trained to take in 512 dim text vectors, we state this in the encoding stage explicitly.
    here we use tokenizer.encode_plus as this can return both the input_ids and attention_masks, otherwise can use tokenizer.encode below, and make your own attention_masks
    """

    max_length = max([len(x) for x in text_corpus])
    input_ids = [
        tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )["input_ids"]
        for x in text_corpus
    ]
    attention_masks = [
        tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )["attention_mask"]
        for x in text_corpus
    ]

    # change into pytorch tensors for model
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_masks)

    # place model and tensors on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids.to(device)
    attention_mask.to(device)

    """ 
    outputs is a tuple with the shape (number of text inputs, max number of tokens in the sequence, number of hidden units in the DistilBERT model).
    """
    print("getting embeddings from BERT model")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    """
    Reshape and turn into numpy array.
    The output tensor shape is [batch_size, sequence_length, hidden_size]. 
    For sentence classification, we’re only only interested in BERT’s output for the [CLS] token,
    so we select that slice of the cube and discard everything else.
    This will be a 768 size vector (the number of hidden units in the DistilBERT model).
    [CLS] token will act as the sentence-embedding, but sometime better to average pool all the word embeddings instead.
    """
    """
    Here we have the option to use mean pooling, which takes the mean of all the output tokens
    hence the sentence representation should hold the meaning of all the words (tokens)
    """
    if use_mean_pooling:
        embedded_features = mean_pooling(outputs, attention_mask)
    else:
        embedded_features = outputs[0][:, 0, :].numpy()
    return embedded_features


def get_embeddings_for_more_than_512(
    tokenizer, model, text_corpus, use_mean_pooling=True
):
    """
    here we don't set a max_length or truncate, as we want to keep all text inputs
    we will proceed with chunking in order to retain all the information
    """
    input_ids = [
        tokenizer.encode_plus(x, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        for x in text_corpus
    ]
    attention_masks = [
        tokenizer.encode_plus(x, add_special_tokens=False, return_tensors="pt")[
            "attention_mask"
        ]
        for x in text_corpus
    ]

    """
    split input_ids and attention_masks into 510 item chunks (add 2 extra [CLS] and [SEP] tokens later)
    this BERT model is trained to take in 512 dim text vectors
    """
    input_ids_chunked = []
    attention_masks_chunked = []
    chunk_ids = []
    count = 0
    # each chunk will contain 510 items
    chunk_size = 510
    for i in range(len(input_ids)):
        input_id_chunks = input_ids[i][0].split(chunk_size)
        attention_mask_chunks = attention_masks[i][0].split(chunk_size)
        for chunk in input_id_chunks:
            input_ids_chunked.append(chunk)
            chunk_ids.append(count)
        for chunk in attention_mask_chunks:
            attention_masks_chunked.append(chunk)
        count = count + 1

    # this will be used later as a scaling factor when re-comibing embeddings
    num_tokens_per_chunk = []
    for input_id_chunk in input_ids_chunked:
        num_tokens_per_chunk.append(len(input_id_chunk))

    """ 
    now add [CLS] and [SEP] tokens and pad the tensors
    can probably use the tokenizer to add special tokens and pad for us here
    """
    for i in range(len(input_ids_chunked)):
        input_id_chunk = input_ids_chunked[i]
        attention_mask_chunk = attention_masks_chunked[i]
        # add [CLS] and [SEP] tokens to tensor
        input_id_chunk = torch.cat(
            [torch.Tensor([101]), input_id_chunk, torch.Tensor([102])]
        )
        # add 1 for [CLS] and [SEP] on attention_masks
        attention_mask_chunk = torch.cat(
            [torch.Tensor([1]), attention_mask_chunk, torch.Tensor([1])]
        )
        # get number of entries to pad
        padding_len = chunk_size + 2 - input_id_chunk.shape[0]
        if padding_len > 0:
            # pad to make all input vectors the same length
            input_id_chunk = torch.cat(
                [input_id_chunk, torch.Tensor([0] * padding_len)]
            )
            attention_mask_chunk = torch.cat(
                [attention_mask_chunk, torch.Tensor([0] * padding_len)]
            )
        # replace all the inputs and masks with the newly tokened and padded vectors
        input_ids_chunked[i] = input_id_chunk
        attention_masks_chunked[i] = attention_mask_chunk

    # stack all the input vectors and format as long and int respectively
    input_ids = torch.stack(input_ids_chunked).long()
    attention_masks = torch.stack(attention_masks_chunked).int()

    # place model and tensors on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids.to(device)
    attention_masks.to(device)

    # pass into model to get embeddings
    print("getting embeddings from BERT model")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
    """
    Reshape and turn into numpy array
    The output tensor shape is [batch_size, sequence_length, hidden_size]. 
    For sentence classification, we’re only only interested in BERT’s output for the [CLS] token,
    so we select that slice of the cube and discard everything else.
    This will be a 768 size vector (the number of hidden units in the DistilBERT model).
    [CLS] token will act as the sentence-embedding, as it contains the aggregated representation of the sentence, 
    but sometime better to average pool all the word embeddings.
    """
    if use_mean_pooling:
        embedded_features = mean_pooling(outputs, attention_masks).numpy()
    else:
        embedded_features = outputs[0][:, 0, :].numpy()

    # creat a dictionary that groups all the chunk_ids and their vector positions
    d = {}
    count = 0
    for i in chunk_ids:
        if i not in d:
            d[i] = []
            d[i].append(count)
        else:
            d[i].append(count)
        count = count + 1

    """
    now re-group the embeddings based on their chunk_ids
    """
    embedding_len = len(embedded_features[0])
    embedded_list = []
    for key, id_list in d.items():
        tot_embedding = np.zeros(embedding_len)
        total_div_factor = 0
        div_factor = 0
        tot_scale_factor = 0
        for embedding_id in id_list:
            embedding = embedded_features[embedding_id]
            # this scaling factor will give a weight to the embedding depending on how much padding was required
            scale_factor = num_tokens_per_chunk[embedding_id] / chunk_size
            # add the embeddings multiplied by the scale factor together
            tot_embedding = tot_embedding + embedding * scale_factor
            # create a total scale factor to normalize at the end
            tot_scale_factor = tot_scale_factor + scale_factor
        # normalize total embedding
        tot_embedding = tot_embedding / tot_scale_factor
        embedded_list.append(tot_embedding)

    embedded_array = np.vstack(embedded_list)

    print("chunked embeddings shape:")
    print(embedded_features.shape)
    print("grouped embeddings shape:")
    print(embedded_array.shape)
    return embedded_array
