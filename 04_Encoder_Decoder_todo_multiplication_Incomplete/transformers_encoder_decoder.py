from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math

from data_loader import get_transformed_text


class SelfAttention(nn.Module):
    def __init__(self, emb_size, output_size, drop_out=0.3):
        super().__init__()
        self.query = nn.Linear(emb_size, output_size)
        self.key = nn.Linear(emb_size, output_size)
        self.value = nn.Linear(emb_size, output_size)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        tgt_len = q.shape[1]
        seq_len = k.shape[1]
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        dim_k = key.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)

        if (mask is not None) and (torch.all(mask).item()):
            expanded_mask = mask[:, None, :].expand(bs, tgt_len, seq_len)
            scores = scores.masked_fill(expanded_mask == 0, -float("inf"))

        weights = F.softmax(scores, dim=-1)
        outputs = torch.bmm(weights, value)
        return outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_output_size = self.emb_size // self.num_heads
        self_att_lst = []
        for _ in range(self.num_heads):
            self_att_lst.append(SelfAttention(emb_size=emb_size, output_size=self.attn_output_size, drop_out=self.dropout))
        self.attentions = nn.ModuleList(self_att_lst)

        self.output = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, q, k, v, mask):
        lst = []
        for head in self.attentions:
            lst.append(head(q, k, v, mask))

        x = torch.cat(lst, dim=-1)
        x = self.output(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, d_ff=2048, dropout=0.3):
        super().__init__()
        # attention
        self.attn = MultiHeadAttention(emb_size, num_heads, dropout=dropout)

        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, emb_size),
            nn.Dropout(dropout),
        )

        # layer norm
        self.attn_norm = nn.LayerNorm(emb_size)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, src, src_mask):
        x = src
        x = self.attn_norm(x)
        x = x + self.attn(q=x, k=x, v=x, mask=src_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, d_ff=2048, dropout=0.3):
        super().__init__()

        # masked attn
        self.masked_attn = MultiHeadAttention(
            emb_size, num_heads, dropout=dropout
        )

        # attn
        self.attn = MultiHeadAttention(
            emb_size, num_heads, dropout=dropout
        )

        ## ffn
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, emb_size),
            nn.Dropout(dropout),
        )

        # layer norm
        self.masked_attn_norm = nn.LayerNorm(emb_size)
        self.attn_norm = nn.LayerNorm(emb_size)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, tgt, enc, tgt_mask, enc_mask):
        x = tgt
        x = self.masked_attn_norm(x)
        x = x + self.masked_attn(q=x, k=x, v=x, mask=tgt_mask)

        x = self.attn_norm(x)
        x = x + self.attn(q=x, k=enc, v=enc, mask=enc_mask)

        x = x + self.ffn(self.ffn_norm(x))
        return x


class Encoder(nn.Module):
    def __init__(self, emb_size, num_heads, num_encoders, d_ff=2048, dropout=0.3):
        super().__init__()
        enc_lst = []
        for _ in range(num_encoders):
            enc_lst.append(EncoderLayer(emb_size=emb_size, num_heads=num_heads, d_ff=d_ff, dropout=dropout))
        self.enc_layers = nn.ModuleList(enc_lst)

    def forward(self, src, src_mask):
        output = src
        for layer in self.enc_layers:
            output = layer(output, src_mask)
        return output


class Decoder(nn.Module):
    def __init__(self, emb_size, num_heads, num_decoders, d_ff=2048, dropout=0.3):
        super().__init__()
        dec_lst = []
        for _ in range(num_decoders):
            dec_lst.append(DecoderLayer(emb_size=emb_size, num_heads=num_heads, d_ff=d_ff, dropout=dropout))
        self.dec_layers = nn.ModuleList(dec_lst)

    def forward(self, tgt, tgt_mask, enc_mask, enc_output):
        output = tgt
        for layer in self.dec_layers:
            output = layer(output, enc_output, tgt_mask, enc_mask)
        return output


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, emb_size=512, num_heads=8, num_encoders=6, num_decoders=6, d_ff=2048, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(emb_size, num_heads, num_encoders, d_ff, dropout)
        self.decoder = Decoder(emb_size, num_heads, num_decoders, d_ff, dropout)

    def forward(self, src, src_mask, tgt, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, tgt_mask, src_mask, enc_out)
        return dec_out

    def decoder_output(self, src, src_mask):
        return self.encoder(src, src_mask)


class TransformerModel(nn.Module):
    """
    In this model a sequence of tokens are passed to the embedding layer first, followed by a positional encoding
    layer to account for the order of the word.

    The nn.TransformerEncoder consists of multiple layers of nn.TransformerEncoderLayer. Along with the input
    sequence, a square attention mask is required because the self-attention layers in nn.TransformerEncoder are
    only allowed to attend the earlier positions in the sequence. For the language modeling task, any tokens on the
     future positions should be masked.

     To produce a probability distribution over output words, the output of the nn.TransformerEncoder model is
     passed through a linear layer followed by a log-softmax function.
    """

    def __init__(self, vocab_size_enc:int, vocab_size_dec:int, emb_size:int, n_head:int,
                 n_layers:int, dropout:float = 0.1, d_ff: int = 2048):
        """
        ntokem: # of unique words in Wikitext2 dataset
        emb_size: embedding vector size
        """
        super().__init__()
        self.model_type = 'Transformer'
        self.emb_size = emb_size
        self.emb_layer_enc = nn.Embedding(vocab_size_enc, emb_size)
        self.emb_layer_dec = nn.Embedding(vocab_size_dec, emb_size)

        self.positional_layer_enc = PositionalEncoding(emb_size, dropout)
        self.positional_layer_dec = PositionalEncoding(emb_size, dropout)

        self.EncoderDecoderTransformer = EncoderDecoderTransformer(emb_size=emb_size,
                                                                   num_heads=n_head,
                                                                   num_encoders=n_layers,
                                                                   num_decoders=n_layers,
                                                                   d_ff=d_ff,
                                                                   dropout=dropout)

        self.classifier = nn.Linear(emb_size, vocab_size_dec) # classifier layer: ntoken would e # of classes

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.emb_layer_enc.weight.data.uniform_(-initrange, initrange)
        self.emb_layer_dec.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor=None, tgt_mask: torch.Tensor=None,
                src_pad_mask: torch.Tensor=None, tgt_pad_mask: torch.Tensor=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.emb_layer_enc(src) * math.sqrt(self.emb_size) # embed the input: shape 35X20Xembedding_size  --> embedding_size=200, [35, Bs] --> [35, Bs, em_size]
        src = self.positional_layer_enc(src)

        tgt = self.emb_layer_dec(tgt) * math.sqrt(self.emb_size)
        tgt = self.positional_layer_enc(tgt)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        decoder_out = self.EncoderDecoderTransformer(src, src_mask, tgt, tgt_mask)

        output = self.classifier(decoder_out)  # linear classification layer: 28782 words/classes -> shape: 35X20X28782
        return output

    def inference_preprocess_enc(self, src: torch.Tensor = None, tgt: torch.Tensor = None):
        if src is not None:
            src = self.emb_layer_enc(src) * math.sqrt(self.emb_size)
            src = self.positional_layer_enc(src)
            src = src.permute(1, 0, 2)

        if tgt is not None:
            tgt = self.emb_layer_dec(tgt) * math.sqrt(self.emb_size)
            tgt = self.positional_layer_enc(tgt)
            tgt = tgt.permute(1, 0, 2)

        return src, tgt


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    """
    - https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
    - https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    - http://jalammar.github.io/illustrated-transformer/
    """
    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_len, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

#####################################################################################

def create_mask(src, tgt, pad_idx, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

#####################################################################################

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    src, _ = model.inference_preprocess_enc(src)
    # enc_out = model.EncoderDecoderTransformer.decoder_output(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        _, ys_embedded = model.inference_preprocess_enc(src=None, tgt=ys)
        # enc_out = enc_out.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        dec_out = model.EncoderDecoderTransformer(src, src_mask, ys_embedded, tgt_mask)
        # dec_out = dec_out.transpose(0, 1)
        # prob = model.classifier(dec_out[:, -1])
        prob = model.classifier(dec_out[:,-1,:])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).type(torch.long).to(device)], dim=0)
        if next_word == end_symbol:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, tgt_language: str, src_language: str, src_sentence: str, special_symbol_idx: Dict,
              tokenizer, vocab, device):
    model.eval()
    text_transform = get_transformed_text(tokenizer, vocab, src_language, tgt_language, special_symbol_idx)
    src = text_transform[src_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5,
                               start_symbol=special_symbol_idx["BOS_IDX"], end_symbol=special_symbol_idx["EOS_IDX"],
                               device=device).flatten()
    lst_translated_tokens = list(tgt_tokens.cpu().numpy())
    lst_translated_words = vocab[tgt_language].lookup_tokens(lst_translated_tokens)
    return " ".join(lst_translated_words).replace("<bos>", "").replace("<eos>", "")