######################################################################
# Seq2Seq Network using Transformer
# ---------------------------------
#
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation tasks.
# Below, we will create a Seq2Seq network that uses Transformer. The network
# consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
# into corresponding tensor of input embeddings. These embedding are further augmented with positional
# encodings to provide position information of input tokens to the model. The second part is the
# actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model.
# Finally, the output of the Transformer model is passed through linear layer
# that gives unnormalized probabilities for each token in the target language.
#

from typing import Dict
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
# from torch_transformer import Transformer
import math

from data_loader import get_transformed_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    # (vocab_size_enc=src_vocab_size, vocab_size_dec=tgt_vocab_size, emb_size=emb_size,
    # n_head=n_head, n_layers=n_layers, dropout=args.dropout, d_ff=args.ffn_hid_dim,
    # src_pad_idx=special_symbol_idx["PAD_IDX"], tgt_psd_idx=special_symbol_idx["PAD_IDX"],
    # max_seq_length=5000, batch_first=args.batch_first, device=device,)

    def __init__(self, vocab_size_enc: int, vocab_size_dec: int,
                 emb_size: int, n_head: int, n_layers: int, dropout: float, d_ff: int,
                 src_pad_idx: int, tgt_psd_idx: int, max_seq_length: int, batch_first: bool, device: str="cpu"):

        super(Seq2SeqTransformer, self).__init__()
        self.batch_first = batch_first
        self.device = device
        self.src_pad_idx = src_pad_idx

        self.transformer = Transformer(d_model=emb_size,
                                       nhead=n_head,
                                       num_encoder_layers=n_layers,
                                       num_decoder_layers=n_layers,
                                       dim_feedforward=d_ff,
                                       dropout=dropout,
                                       batch_first=self.batch_first)

        self.src_tok_emb = TokenEmbedding(vocab_size_enc, emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size_dec, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.classification = nn.Linear(emb_size, vocab_size_dec)

    def forward(self, X_src: Tensor, X_tgt: Tensor,):
        """
        X_src shape: (src_sent_len, Bs)
        X_tgt shape: (tgt_sent_len, Bs)
        """
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(X_src, X_tgt, self.src_pad_idx, self.device)
        memory_key_padding_mask = src_padding_mask

        if self.batch_first:
            X_src = X_src.permute(1, 0)
            X_tgt = X_tgt.permute(1, 0)
        X_src_emb = self.positional_encoding(self.src_tok_emb(X_src))
        X_tgt_emb = self.positional_encoding(self.tgt_tok_emb(X_tgt))
        X_outs = self.transformer(X_src_emb, X_tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.classification(X_outs)

    def encode(self, X_src: Tensor, src_mask: Tensor):
        pe = self.positional_encoding(self.src_tok_emb(X_src))
        return self.transformer.encoder(pe, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        pe = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(pe, memory, tgt_mask)


######################################################################
# During training, we need a subsequent word mask that will prevent the model from looking into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both.
#


def generate_square_subsequent_mask(sz):
    """
    sz = sent_length.
    For example, for a sent_length=3:
                    [[0, -inf, -inf],
                     [0,   0,  -inf],
                     [0,   0,    0 ]]
    """
    ones = torch.ones((sz, sz), device=device) 
    triu = torch.triu(ones)  # For instance: [[1, 1, 1],
    #                                         [0, 1, 1],
    #                                         [0, 0, 1]]

    mask1 = (triu == 1).transpose(0, 1) # For instance: [[T, F, F],
    #                                                    [T, T, F],
    #                                                    [T, T, T]]
    mask1 = mask1.float().masked_fill(mask1 == False, float('-inf'))
    mask = mask1.masked_fill(mask1 == 1, float(0.0))   # For instance: [[0, -inf, -inf],
    #                                                                   [0,   0,  -inf],
    #                                                                   [0,   0,    0 ]]
    return mask


def create_mask(src, tgt, pad_idx, device="cpu"):
    """
    X_src shape: (src_sent_len, Bs)
    X_tgt shape: (tgt_sent_len, Bs)
    """    
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # While in the X_target (decoder) we want each word only get attention from previous words, in X_src each word get identical
    # attentions from all previous and subsequnt (future) words. That is why src_mask is a zero matrix.
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)                  # tgt_mask shape: (tgt_seq_len, tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  # src_mask shape: (src_seq_len, src_seq_len)

    src_padding_mask = (src == pad_idx).transpose(0, 1) # src_padding_mask shape: (Bs, src_sent_len)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1) # tgt_padding_mask shape: (Bs, tgt_sent_len)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

#####################################################################################

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.classification(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    return ys


# actual function to translate input sentence into target language
def translate_torch_tutorial(model: torch.nn.Module, tgt_language: str, src_language: str, src_sentence: str,
                             special_symbol_idx: Dict, tokenizer, vocab, device,
                             beam_search_generation_approach, beam_size, n_results, max_length):
    model.eval()
    text_transform = get_transformed_text(tokenizer, vocab, src_language, tgt_language, special_symbol_idx)
    src = text_transform[src_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5,
                               start_symbol=special_symbol_idx["BOS_IDX"], end_symbol=special_symbol_idx["EOS_IDX"],
                               device=device).flatten()
    return " ".join(vocab[tgt_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")