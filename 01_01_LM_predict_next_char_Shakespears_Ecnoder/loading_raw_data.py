import torch
import os

torch.manual_seed(1337)


def get_train_test_data(file_path='input.txt'):
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        from urllib.request import urlopen
        url_target = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = urlopen(url_target).read().decode("utf-8")

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)} # stoi: string to int
    itos = {i: ch for i, ch in enumerate(chars)} # itos: int to string
    encode_func = lambda s: [stoi[c] for c in s] # encoder function: take a string, output a list of integers
    decode_func = lambda l: ''.join([itos[i] for i in l]) # decoder function: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode_func(text), dtype=torch.long) # data in int-long torch tensor: shape: [1115394]
    n = int(0.9*len(data)) # first 90% will be used for training, rest for val
    train_data = data[:n]
    val_data = data[n:]
    return encode_func, decode_func, vocab_size, train_data, val_data
