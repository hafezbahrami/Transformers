from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
import math

from data_loader import get_transformed_text


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, output_emb_size, drop_out=0.3, batch_first=False):
        super().__init__()
        self.query = nn.Linear(emb_size, output_emb_size, bias=False)
        self.key = nn.Linear(emb_size, output_emb_size, bias=False)
        self.value = nn.Linear(emb_size, output_emb_size, bias=False)
        self.dropout = nn.Dropout(drop_out)
        self.batch_first = batch_first

    def forward(self, q, k, v, mask=None):
        """The code below is with the assumption that batch-size-first is False"""
        if not self.batch_first:
            """We need q, k, v to be in batch_size first shape"""
            q, k, v = q.transpose(1, 0), k.transpose(1, 0), v.transpose(1, 0)
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        emb_size = key.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(emb_size)

        # below code requires debugging. It creates error as of now
        if mask is not None:
            # mask should have shape (T, T). All zero's (or False's) will be "-inf"
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        outputs = torch.bmm(weights, value)
        return outputs.transpose(1, 0) if not self.batch_first else outputs# again batch-first flag is False


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, batch_first):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.attn_output_emb_size = self.emb_size // self.num_heads
        self_att_lst = []
        for _ in range(self.num_heads):
            self_att_lst.append(SingleHeadSelfAttention(emb_size=emb_size, output_emb_size=self.attn_output_emb_size,
                                                        drop_out=dropout, batch_first=batch_first))
        self.attention_heads = nn.ModuleList(self_att_lst)

        self.proj = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        lst = []
        for head in self.attention_heads:
            lst.append(head(q=q, k=k, v=v, mask=mask))

        x = torch.cat(lst, dim=-1)
        x = self.dropout(self.proj(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, d_ff=2048, dropout=0.3, batch_first=False):
        super().__init__()
        # attention
        self.attn = MultiHeadAttention(emb_size=emb_size, num_heads=num_heads, dropout=dropout, batch_first=batch_first)

        # ffwd
        self.ffwd = FeedFoward(emb_size=emb_size, d_ff=d_ff, dropout=dropout)

        # layer norm
        self.attn_norm = LayerNorm(emb_size, eps=1e-5, bias=True)
        self.ffwd_norm = LayerNorm(emb_size, eps=1e-5, bias=True)

        self.norm_first=False

    def forward(self, src, src_mask, src_padding_mask):
        mask = self._process_mask(src, src_mask, src_padding_mask)
        x = src
        if self.norm_first:
            x = self.attn_norm(x)
            x = src + self.attn(q=x, k=x, v=x, mask=mask)
            x = x + self.ffwd(self.ffwd_norm(x))
        else:
            x = src + self.attn(q=x, k=x, v=x, mask=mask)
            x = self.attn_norm(x)

            x = x + self.ffwd(x)
            x = self.ffwd_norm(x)

        return x

    def _process_mask(self, src, src_mask, src_padding_mask):

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        src_padding_mask = F._canonical_mask(
            mask=src_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="mask",
            target_type=src.dtype
        )
        src_mask = src_mask[None, :, :]
        src_padding_mask = src_padding_mask[:, None, :]
        mask = src_mask + src_padding_mask
        return mask


class DecoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, d_ff=2048, dropout=0.3, batch_first=False):
        super().__init__()

        # masked attn
        self.masked_attn = MultiHeadAttention(
            emb_size, num_heads, dropout=dropout, batch_first=batch_first
        )

        # attn
        self.attn = MultiHeadAttention(
            emb_size, num_heads, dropout=dropout, batch_first=batch_first
        )

        # ffwd
        self.ffwd = FeedFoward(emb_size=emb_size, d_ff=d_ff, dropout=dropout)

        # layer norm
        self.masked_attn_norm = LayerNorm(emb_size)
        self.cross_attn_norm = LayerNorm(emb_size)
        self.ffwd_norm = LayerNorm(emb_size)

        self.norm_first = False

    def forward(self, tgt, Y_enc_out, tgt_mask, src_padding_mask, tgt_padding_mask):
        self_attn_mask = self._process_mask(tgt, tgt_mask, tgt_padding_mask)
        cross_head_attn_mask = self._process_mask(tgt, None, src_padding_mask)

        y = tgt
        if self.norm_first:
            y = self.masked_attn_norm(y)
            y = tgt + self.masked_attn(q=y, k=y, v=y, mask=self_attn_mask)

            # cross_attention (memory attention)
            y = y + self.attn(q=self.cross_attn_norm(y), k=Y_enc_out, v=Y_enc_out, mask=cross_head_attn_mask)
            y = y + self.ffwd(self.ffwd_norm(y))
        else:
            y = tgt + self.masked_attn(q=y, k=y, v=y, mask=self_attn_mask)
            y = self.masked_attn_norm(y)

            # cross_attention (memory attention)
            y = y + self.attn(q=y, k=Y_enc_out, v=Y_enc_out, mask=cross_head_attn_mask)
            y = self.cross_attn_norm(y)
            y = y + self.ffwd(y)
            y = self.ffwd_norm(y)

        return y

    def _process_mask(self, src, src_mask, src_padding_mask):

        src_padding_mask = F._canonical_mask(
            mask=src_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="mask",
            target_type=src.dtype
        )
        if src_mask is None:
            return src_padding_mask[:, None, :]

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        src_mask = src_mask[None, :, :]
        src_padding_mask = src_padding_mask[:, None, :]
        mask = src_mask + src_padding_mask
        return mask


class Encoder(nn.Module):
    def __init__(self, emb_size, num_heads, num_encoders, d_ff=2048, dropout=0.3, batch_first=False):
        super().__init__()
        enc_lst = []
        for _ in range(num_encoders):
            enc_lst.append(EncoderBlock(emb_size=emb_size, num_heads=num_heads, d_ff=d_ff, dropout=dropout, batch_first=batch_first))
        self.enc_block = nn.ModuleList(enc_lst)

    def forward(self, src, src_mask, src_padding_mask):
        output = src
        for layer in self.enc_block:
            output = layer(output, src_mask, src_padding_mask)
        return output


class Decoder(nn.Module):
    def __init__(self, emb_size, num_heads, num_decoders, d_ff=2048, dropout=0.3, batch_first=False):
        super().__init__()
        dec_lst = []
        for _ in range(num_decoders):
            dec_lst.append(DecoderBlock(emb_size=emb_size, num_heads=num_heads, d_ff=d_ff, dropout=dropout, batch_first=batch_first))
        self.dec_blocks = nn.ModuleList(dec_lst)

    def forward(self, tgt, Y_enc_out, tgt_mask, src_padding_mask, tgt_padding_mask):
        output = tgt
        for layer in self.dec_blocks:
            output = layer(output, Y_enc_out, tgt_mask, src_padding_mask, tgt_padding_mask)
        return output


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, emb_size, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, emb_size=512, num_heads=8, num_encoders=6, num_decoders=6, d_ff=2048, dropout=0.3,
                 batch_first=False):
        super().__init__()
        self.encoder = Encoder(emb_size, num_heads, num_encoders, d_ff, dropout, batch_first)
        self.decoder = Decoder(emb_size, num_heads, num_decoders, d_ff, dropout, batch_first)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        enc_out = self.encoder(src, src_mask, src_padding_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_padding_mask, tgt_padding_mask)
        return dec_out

    def encoder_output(self, src, src_mask):
        return self.encoder(src, src_mask)


class TransformerModel(nn.Module):
    """
    In this model a sequence of tokens are passed to the embedding layer first, followed by a positional encoding
    layer to account for the order of the word.

    The nn.TransformerEncoder consists of multiple layers of nn.TransformerEncoderBlock. Along with the input
    sequence, a square attention mask is required because the self-attention layers in nn.TransformerEncoder are
    only allowed to attend the earlier positions in the sequence. For the language modeling task, any tokens on the
     future positions should be masked.

     To produce a probability distribution over output words, the output of the nn.TransformerEncoder model is
     passed through a linear layer followed by a log-softmax function.
    """

    def __init__(self, vocab_size_enc: int, vocab_size_dec: int, emb_size: int, n_head: int,
                 n_layers: int, dropout: float = 0.1, d_ff: int = 2048, src_pad_idx: int = 1, tgt_psd_idx: int = 1,
                 max_seq_length: int = 5000, batch_first: bool = False, device="cpu"):
        """
        vocab_size_enc: # of unique words in Wikitext2 dataset for source language (German)
        emb_size: embedding vector size
        rest: self-explanatory
        """
        super().__init__()
        self.model_type = 'Transformer'

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_psd_idx
        self.max_seq_length = max_seq_length
        self.device = device
        self.batch_first = batch_first

        self.emb_size = emb_size
        self.emb_layer_enc = TokenEmbedding(vocab_size_enc, emb_size)
        self.emb_layer_dec = TokenEmbedding(vocab_size_dec, emb_size)

        self.positional_encoding = PositionalEncoding(emb_size, dropout, self.max_seq_length)

        self.EncoderDecoderTransformer = EncoderDecoderTransformer(emb_size=emb_size, num_heads=n_head,
                                                                   num_encoders=n_layers, num_decoders=n_layers,
                                                                   d_ff=d_ff, dropout=dropout,
                                                                   batch_first=batch_first)

        self.classifier = nn.Linear(emb_size, vocab_size_dec) # classifier layer: vocab_size_dec would be # of classes

        # self.init_weights()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.emb_layer_enc.weight.data.uniform_(-initrange, initrange)
    #     self.emb_layer_dec.weight.data.uniform_(-initrange, initrange)
    #     self.classifier.bias.data.zero_()
    #     self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Args:
            src: Tensor, shape [src_seq_len, batch_size]
            tgt: Tensor, shape [tgt_seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # For debugging purpose
        # src = torch.tensor([[9, 6, 1, 1],
        #                     [9, 5, 2, 1]]).permute(1, 0)
        # tgt = torch.tensor([[9, 6, 5, 5, 1, 1],
        #                     [9, 5, 5, 5, 2, 1]]).permute(1, 0)

        src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._readiness_for_transformer(src, tgt)

        decoder_out = self.EncoderDecoderTransformer(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

        output = self.classifier(decoder_out)  # linear classification layer: 28782 words/classes -> shape: 35X20X28782
        return output

    def _readiness_for_transformer(self, src: torch.Tensor = None, tgt: torch.Tensor = None,
                                   src_embedding: bool = True, tgt_embedding: bool = True):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, self.src_pad_idx, self.device)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        if self.batch_first:
            src = src.permute(1, 0)
            tgt = tgt.permute(1, 0)
        if (src is not None) and src_embedding:
            src = self.emb_layer_enc(src) # embed the input: shape 35X20Xembedding_size  --> embedding_size=200, [35, Bs] --> [35, Bs, em_size]
            src = self.positional_encoding(src)

        if (tgt is not None) and tgt_embedding:
            tgt = self.emb_layer_dec(tgt)
            tgt = self.positional_encoding(tgt)

        return src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    """
    - https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
    - https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    - http://jalammar.github.io/illustrated-transformer/
    """
    # def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 5000):
    #     super().__init__()
    #     self.dropout = nn.Dropout(p=dropout)

    #     position = torch.arange(max_len).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
    #     pe = torch.zeros(max_len, 1, emb_size)
    #     pe[:, 0, 0::2] = torch.sin(position * div_term)
    #     pe[:, 0, 1::2] = torch.cos(position * div_term)
    #     self.register_buffer('pe', pe)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         x: Tensor, shape [seq_len, batch_size, embedding_dim]
    #     """
    #     x = x + self.pe[:x.size(0)]
    #     return self.dropout(x)
    
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

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])    

#####################################################################################
def generate_square_subsequent_mask(sz: int, device="cpu") -> torch.Tensor:
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
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device).to(device)                  # tgt_mask shape: (tgt_seq_len, tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  # src_mask shape: (src_seq_len, src_seq_len)

    src_padding_mask = (src == pad_idx).transpose(0, 1) # src_padding_mask shape: (Bs, src_sent_len)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1) # tgt_padding_mask shape: (Bs, tgt_sent_len)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

#####################################################################################


# Beam Search
class TopNHeap:
    """
    A heap that keeps top N elements, in forms of (float, str).
    Example: [(-1, "I am student), (-10, "I am blackboard), ...]
    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, new_el):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, new_el)
        elif new_el > self.elements[0]:
            heapreplace(self.elements, new_el)


def beam_search(model: TransformerModel, src, src_mask,
                start_symbol, end_symbol, vocab, text_transform,
                beam_size: int, n_results: int = 10, max_length: int = 30,
                average_log_liklihood: bool = False,
                device = "cpu"):
    results = TopNHeap(n_results)

    beam = TopNHeap(beam_size)
    beam.add((0, ""))

    src_no_embed = src.to(device)
    for iter in range(max_length):
        new_beam = TopNHeap(beam_size)

        for log_score, sent in beam.elements:
            # predict next word
            tgt = text_transform(sent).view(-1, 1).to(device) # get it in tensor format
            src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model._readiness_for_transformer(
                                                                                                    src_no_embed, tgt)
            dec_out = model.EncoderDecoderTransformer(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

            dec_out2 = dec_out[:, -1, :] if model.batch_first else dec_out[-1, :, :]
            logits = model.classifier(dec_out2)

            ls = torch.nn.functional.log_softmax(logits, dim=-1).view(-1).detach()

            for i, word in enumerate(vocab.get_itos()):
                new_sent = sent + " " + word
                if word == ".":
                    norm = 1. / len(sent+word) if average_log_liklihood else 1
                    results.add(((log_score + float(ls[i]))*norm, new_sent))
                else:
                    new_beam.add((log_score + float(ls[i]), new_sent))
        beam = new_beam
    results_elem = [(l, s) for l, s in results.elements]
    results_elem.sort(key=lambda x: x[0], reverse=True)
    return results_elem[0][1]

#####################################################################################


# function to generate translated sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    src_no_embed = src.to(device)
    tgt_no_embed = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model._readiness_for_transformer(
                                                                                            src_no_embed, tgt_no_embed)
        dec_out = model.EncoderDecoderTransformer(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

        dec_out2 = dec_out[:, -1, :] if model.batch_first else dec_out[-1, :, :]
        prob = model.classifier(dec_out2)

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        tgt_no_embed = torch.cat([tgt_no_embed, torch.ones(1, 1).fill_(next_word).type(torch.long).to(device)], dim=0)
        if next_word == end_symbol:
            break
    return tgt_no_embed


# function to translate input sentence into target language, or simply to translate
def translate(model: torch.nn.Module, tgt_language: str, src_language: str, src_sentence: str, special_symbol_idx: Dict,
              tokenizer, vocab, device, beam_search_generation_approach, beam_size, n_results, max_length):
    model.eval()
    text_transform = get_transformed_text(tokenizer, vocab, src_language, tgt_language, special_symbol_idx)
    src = text_transform[src_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    if not beam_search_generation_approach:
        tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5,
                                   start_symbol=special_symbol_idx["BOS_IDX"], end_symbol=special_symbol_idx["EOS_IDX"],
                                   device=device).flatten()
        tgt_tokens = tgt_tokens.cpu().numpy()
        lst_translated_tokens = list(tgt_tokens)
        lst_translated_words = vocab[tgt_language].lookup_tokens(lst_translated_tokens)
        translated_sent = " ".join(lst_translated_words)
    else:
        translated_sent = beam_search(model=model, src=src, src_mask=src_mask,
                                    start_symbol=special_symbol_idx["BOS_IDX"], end_symbol=special_symbol_idx["EOS_IDX"],
                                    vocab=vocab[tgt_language], text_transform=text_transform[tgt_language],
                                    beam_size=beam_size, n_results=n_results, max_length=max_length,
                                    average_log_liklihood=False,
                                    device=device)

    return translated_sent.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "")