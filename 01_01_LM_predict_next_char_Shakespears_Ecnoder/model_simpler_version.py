import torch
import torch.nn as nn
from torch.nn import functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1337)


# super simple bigram model
class BigramLanguageModel(nn.Module):
    """A BiGram Language model using the Transformer architecture"""
    def __init__(self, vocab_size, emb_size, sent_len, n_head, n_layer, dropout, sinusoidal_pos_encoding=True):
        super().__init__()
        self.sinusoidal_pos_encoding = sinusoidal_pos_encoding
        self.sent_len = sent_len
        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        if sinusoidal_pos_encoding:
            self.position_embedding_table = PositionalEncoding(max_len=sent_len, emb_size=emb_size, dropout=dropout)
        else:
            self.position_embedding_table = nn.Embedding(sent_len, emb_size)
        l_trans_blk = [TransformerBlock(emb_size, sent_len, n_head, dropout) for _ in range(n_layer)]
        self.transformer_blocks = nn.Sequential(*l_trans_blk)
        self.ln_f = nn.LayerNorm(emb_size) # final layer norm
        self.lm_head_classifier = nn.Linear(emb_size, vocab_size)

    def forward(self, X, targets=None):
        Bs, T = X.shape # T: sent_length
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(X) # (Bs,T,emb_size)
        if self.sinusoidal_pos_encoding:
            pos_emb = self.position_embedding_table(tok_emb)  # (T,emb_size)
        else:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,emb_size)
        x = tok_emb + pos_emb # with broadcasting (right-alighned, then add 1 for the missing dimension) ==> (Bs,T,emb_size)
        x = self.transformer_blocks(x) # (Bs,T,emb_size)
        x = self.ln_f(x) # (Bs,T,emb_size)
        logits = self.lm_head_classifier(x) # (Bs,T,vocab_size)

        if targets is None:
            loss = None
        else:
            """Getting the shape in the format the nn.CrossEntropyLoss() needs"""
            Bs, T, c_vocab_size = logits.shape # (Bs,T,emb_size)
            logits = logits.view(Bs*T, c_vocab_size)
            targets = targets.view(Bs*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, X, max_len_sentence_to_be_created):
        # X is (Bs, T) array of indices in the current context
        for _ in range(max_len_sentence_to_be_created):
            X_context = X[:, -self.sent_len:] # crop X to the last sent_len tokens
            logits, loss = self(X_context) # get the predictions
            logits = logits[:, -1, :] # becomes (Bs, emb_size): focus only on the last time step
            probs = F.softmax(logits, dim=-1) # (Bs, emb_size): apply softmax to get probabilities
            X_next = torch.multinomial(probs, num_samples=1) # (B, 1): sample from the distribution
            X = torch.cat((X, X_next), dim=1) # (Bs, T+1): append sampled index to the running sequence
        return X


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

class SingleHeadSelfAttention(nn.Module):
    """ one head of self-attention """
    def __init__(self, emb_size, sent_len, n_head, dropout):
        super().__init__()
        if emb_size % n_head != 0:
            emb_size = (emb_size // n_head) * n_head
            print("embed size is changes to have a compatible head numbers in self_attention within the transformer.")

        head_emb_size = emb_size // n_head
        self.emb_size = emb_size
        self.key = nn.Linear(emb_size, head_emb_size, bias=False)
        self.query = nn.Linear(emb_size, head_emb_size, bias=False)
        self.value = nn.Linear(emb_size, head_emb_size, bias=False)
        # tril is not parameter. In pytorch, then we should assign it to the Module using register_buffer
        self.register_buffer('tril', torch.tril(torch.ones(sent_len, sent_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Bs, T, a = x.shape # Bs: BatchSize, T: sent_length, Embed_size
        k = self.key(x)   # (Bs,T,head_embed_size)
        q = self.query(x) # (Bs,T,head_embed_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * self.emb_size**-0.5 # C=head_embed_size => (Bs, T, C) @ (Bs, C, T) -> (Bs, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (Bs, T, T)
        wei = F.softmax(wei, dim=-1) # (Bs, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (Bs,T,head_embed_size)
        out = wei @ v # C=head_embed_size => (Bs, T, T) @ (Bs, T, C) -> (Bs, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, emb_size, sent_len, n_head, dropout):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadSelfAttention(emb_size, sent_len, n_head, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # out: Bs X sent_len X (head_emb_size + head_emb_size + head_emb_size+ ...)
        out = self.dropout(self.proj(out)) # This is why emb_size should be divisible by n_heads
        return out # Thanks to self.proj layer, we get the same size embed after each mutiHeadAttention => out: Bs X sent_len X emb_size


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, emb_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, emb_size, sent_len, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(emb_size, sent_len, n_head, dropout) # sa = self-attention
        self.ffwd = FeedFoward(emb_size, dropout)
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.layer_norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.sa(self.layer_norm1(x)) # first residual connection  ==> sa = self-attention
        x = x + self.ffwd(self.layer_norm2(x)) # second residual connection
        return x

