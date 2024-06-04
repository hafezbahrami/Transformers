import torch
import torch.nn.functional as F

def single_head_attention(query, key, value, causal_mask=None, padding_mask=None):
    batch_size, seq_length, d_model = query.size()
    
    # Scaled dot-product attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_model ** 0.5)
    
    if causal_mask is not None:
        # causal_mask should have shape (T, T)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    
    if padding_mask is not None:
        # padding_mask should have shape (B, T)
        padding_mask = padding_mask.unsqueeze(1).expand(batch_size, seq_length, seq_length)
        scores = scores.masked_fill(padding_mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, value)
    
    return context, attn_weights

# Encoder Self-Attention
def encoder_self_attention(X_encoder, src_padding_mask=None):
    return single_head_attention(query=X_encoder, key=X_encoder, value=X_encoder, padding_mask=src_padding_mask)

# Decoder Self-Attention with causal mask
def decoder_self_attention(X_decoder, tgt_padding_mask=None):
    batch_size, tgt_length, d_model = X_decoder.size()
    causal_mask = torch.tril(torch.ones(tgt_length, tgt_length)).bool()
    return single_head_attention(query=X_decoder, key=X_decoder, value=X_decoder, causal_mask=causal_mask, padding_mask=tgt_padding_mask)

# Encoder-Decoder Attention
def enc_dec_cross_attention(X_decoder, Y_encoder, src_padding_mask=None):
    """It seems that for cross attention, key and value comes form Y_encoder. Also, only we consider the padding from the encoder"""
    return single_head_attention(query=X_decoder, key=Y_encoder, value=Y_encoder, padding_mask=src_padding_mask)


# ---------------------------------------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------------------------------------
batch_size, src_length, tgt_length, d_model =  2, 3, 4, 1       # (B, T_s, T_t, C)

# Random encoder and decoder inputs
X_encoder = torch.randn(batch_size, src_length, d_model)    # (B, T_s, C)
X_decoder = torch.randn(batch_size, tgt_length, d_model)    # (B, T_t, C)

# Example src_padding_mask  (Encoder)
src_padding_mask = torch.randint(0, 2, (batch_size, src_length)).bool() # (B, T_s)

# Example tgt_padding_mask  (Decoder)
tgt_padding_mask = torch.randint(0, 2, (batch_size, tgt_length)).bool() # (B, T_t)

# Encoder Self-Attention: No need for causal positional masking (all words gets attention from previous and after words)
Y_encoder, encoder_self_attn_weights = encoder_self_attention(X_encoder, src_padding_mask)

# Decoder Self-Attention
Y_decoder, decoder_self_attn_weights = decoder_self_attention(X_decoder, tgt_padding_mask)

# Encoder-Decoder Attention
context, enc_dec_attn_weights = enc_dec_cross_attention(Y_decoder, Y_encoder, src_padding_mask)

print("Encoder output shape:", Y_encoder.shape)
print("Decoder output shape:", Y_decoder.shape)
print("Context shape (from encoder-decoder attention):", context.shape)
