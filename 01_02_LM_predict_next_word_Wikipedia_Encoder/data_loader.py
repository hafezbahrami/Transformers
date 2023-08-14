# torchdata is in Beta release and may not be stable. Future API might change
# https://github.com/pytorch/data

"""
The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified
 Good and Featured articles on Wikipedia. The WikiText-2 dataset is a small version of the WikiText-103 dataset as it
 contains only 2 million tokens.
"""

from typing import Tuple, List

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
from torch import Tensor
from torch.utils.data import dataset

print_sample_data = False

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def get_transformed_text(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """
    Converts raw text into a flat Tensor. -> Vocab( tokenizer("I am a student) ) = [64, 1731, 8, 1674]
    Putting all the sentences together (concat) and return them all as one giant int/long tensor.

    returns --> torch.tensor([]), depending on the input text, but it could get to length of millions
    """
    if print_sample_data:
        a = [sent for sent in raw_text_iter]
        print(a[:10])
    # Tokenize each word at each sentence, then convert it to one-hot-vector using vocab object
    # data = list( tensor_sent_1, tensor_sen_2, ...)
    data = [torch.tensor(vocab(tokenizer(sent)), dtype=torch.long) for sent in raw_text_iter]
    # filter our data with zero length, then put all sentences after each other (torch.cat)
    lst_of_sent_tensors: List[Tensor] = tuple( filter(lambda t: t.numel() > 0, data) )
    all_sents_in_one_tensor = torch.cat(lst_of_sent_tensors) # shape = [~2,000,000] for training set
    return all_sents_in_one_tensor


def batchify(data: Tensor, bsz: int, device: torch.device, batch_size_first: bool) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
        device: torch.device
        batch_size_first: boolean to have the Bs as first dimension

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]  # making sure to only keep a multiplier of the batch-size. Trim-off rest of sentences.
    if batch_size_first:
        data = data.view(bsz, seq_len).contiguous()
    else:
        data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch_x_y_sents(source_data: Tensor, i: int, bptt: int, batch_size_first: bool) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source_data: Tensor, shape [full_sent_len_in_a_batch, batch_size]
        i: int
        bptt: int ==> sentence length (or sometimes is called block-size) => for instance sentence length = 35
        batch_size_first: bool  ==> If the source data is coming with batch size first

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    raw_data_len = source_data.size(1) if batch_size_first else source_data.size(0)

    seq_len = min(bptt, raw_data_len-1-i)

    if batch_size_first:
        x_data = source_data[:, i: i + seq_len]
        y_target = source_data[:, i+1: i+seq_len+1] # The reshaping part is defined in the loss function
    else:
        x_data = source_data[i: i+seq_len, :]
        y_target = source_data[i+1: i+seq_len+1, :].reshape(-1)

    return x_data, y_target


def get_train_test_eval_data(batch_size, eval_batch_size, device: torch.device, batch_size_first=False):
    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    # get all sentences tokenized, look up at the vocab to numeric value of each word, and then concat them all in one
    # giant int tensor
    train_data: Tensor = get_transformed_text(train_iter) # shape = [~2,000,000] for training set
    val_data: Tensor = get_transformed_text(val_iter)  # shape = [~200, 000]
    test_data: Tensor = get_transformed_text(test_iter)  # shape = [~200, 000]

    train_data_batched = batchify(train_data, batch_size, device, batch_size_first)  # shape [seq_len, batch_size], if the batch_size_first is False
    val_data_batched = batchify(val_data, eval_batch_size, device, batch_size_first)
    test_data_batched = batchify(test_data, eval_batch_size, device, batch_size_first)

    return train_data_batched, val_data_batched, test_data_batched

