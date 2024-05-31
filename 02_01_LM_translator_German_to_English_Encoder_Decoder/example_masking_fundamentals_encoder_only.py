import torch
import torch.nn.functional as F

Only_Encoder: False # This is for LLM, that is only for text generation, and we need to mask-out the future positions in the X_src,
                    # as input to the encoder

def single_head_attention(query, key, value, src_mask=None, src_padding_mask=None):
    batch_size, seq_length, d_model = query.size() # (B, T, C)
    
    # Scaled dot-product attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_model ** 0.5) # we_0: (B, T, T)
    
    if src_mask is not None:
        # src_mask should have shape (T, T). All zero's (or False's) will be "-inf"
        scores = scores.masked_fill(src_mask == 0, float('-inf'))
    
    if src_padding_mask is not None:
        # original src_padding_mask is in shape (B, T)
        src_padding_mask = src_padding_mask.unsqueeze(1).expand(batch_size, seq_length, seq_length)
        scores = scores.masked_fill(src_padding_mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, value)
    
    return context, attn_weights


# ---------------------------------------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------------------------------------
batch_size, seq_length, d_model = 2, 3, 1

query = torch.randn(batch_size, seq_length, d_model)    # (B, T, C)
key = torch.randn(batch_size, seq_length, d_model)      # (B, T, C)
value = torch.randn(batch_size, seq_length, d_model)    # (B, T, C)

# Example src_mask (seq_length, seq_length) for causal masking. Mask out the future positions (zer'ing/False'ing future positions)
src_mask = torch.tril(torch.ones(seq_length, seq_length)).bool()            # (T, T)

# Example src_padding_mask (batch_size, seq_length) for padding masking
src_padding_mask = torch.randint(0, 2, (batch_size, seq_length)).bool()     # (B, T)

# Call the single-head attention function
context, attn_weights = single_head_attention(query, key, value, src_mask, src_padding_mask)

print("Context shape:", context.shape)
print("Attention weights shape:", attn_weights.shape)