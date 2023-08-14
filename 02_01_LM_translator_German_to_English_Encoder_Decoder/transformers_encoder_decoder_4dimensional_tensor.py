import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, n_head):
        super(MultiHeadAttention, self).__init__()
        assert emb_size % n_head == 0, "emb_size must be divisible by n_head"

        self.emb_size = emb_size
        self.n_head = n_head
        self.d_k = emb_size // n_head

        self.W_q = nn.Linear(emb_size, emb_size)
        self.W_k = nn.Linear(emb_size, emb_size)
        self.W_v = nn.Linear(emb_size, emb_size)
        self.W_o = nn.Linear(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, emb_size = x.size()
        return x.view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.emb_size)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_size, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_size, d_ff)
        self.fc2 = nn.Linear(d_ff, emb_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, emb_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, emb_size, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(emb_size, n_head)
        self.feed_forward = PositionWiseFeedForward(emb_size, d_ff)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, emb_size, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(emb_size, n_head)
        self.cross_attn = MultiHeadAttention(emb_size, n_head)
        self.feed_forward = PositionWiseFeedForward(emb_size, d_ff)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class TransformerModel4Dimensional(nn.Module):
    def __init__(self, vocab_size_enc: int, vocab_size_dec: int, emb_size: int, n_head: int,
                    n_layers: int, dropout: float = 0.1, d_ff: int = 2048, src_pad_idx: int = 1, tgt_psd_idx: int = 1,
                    max_seq_length: int = 5000, device="cpu"):
        """
        variables are self-explanatory
        """
        super(TransformerModel4Dimensional, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_psd_idx = tgt_psd_idx
        self.encoder_embedding = nn.Embedding(vocab_size_enc, emb_size)
        self.decoder_embedding = nn.Embedding(vocab_size_dec, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(emb_size, n_head, d_ff, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(emb_size, n_head, d_ff, dropout) for _ in range(n_layers)])

        self.fc = nn.Linear(emb_size, vocab_size_dec)
        self.dropout = nn.Dropout(dropout)

        self.device = device

    def generate_mask(self, src, tgt):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        tgt_mask = (tgt != self.tgt_psd_idx).unsqueeze(1).unsqueeze(3).to(self.device)
        # src_mask = (src != 0).unsqueeze(1)
        # tgt_mask = (tgt != 0).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)  # we could have directly used => torch.tril(torch.ones(tgt_len, tgt_len))
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # make it Batch-size first
        src = src.permute(1, 0).to(self.device)
        tgt = tgt.permute(1, 0).to(self.device)
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

if __name__ == "__main__":
    # Testing the Transformer code above
    vocab_size_enc = 5000
    vocab_size_dec = 5000
    emb_size = 6
    n_head = 1
    n_layers = 1
    d_ff = 24
    max_seq_length = 100
    dropout = 0.1
    src_pad_idx = 1
    tgt_psd_idx = 1
    
    transformer = TransformerModel4Dimensional(vocab_size_enc, vocab_size_dec, emb_size, n_head, n_layers, dropout, d_ff,
                                   src_pad_idx, tgt_psd_idx, max_seq_length)

    # Generate random sample data: Bs=64
    # src_data = torch.randint(1, vocab_size_enc, (64, max_seq_length))  # (Bs, seq_length)
    # tgt_data = torch.randint(1, vocab_size_dec, (64, max_seq_length))  # (Bs, seq_length)
    
    pad_index = 0
    src_data = torch.tensor([[9, 6, pad_index, pad_index],
                             [9, 5, 2, pad_index]])
    tgt_data = torch.tensor([[9, 6, 5, 5, pad_index, pad_index],
                             [9, 5, 5, 5, 2, pad_index]])
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    transformer.train()
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, vocab_size_dec), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")






