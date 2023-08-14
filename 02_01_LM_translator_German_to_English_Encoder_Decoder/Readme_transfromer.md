# Transformer Details

# Note:
For now, the Transformer model from Pytorch Tutorial works fine, the Transfromer model I wrote form scratch does not get tranined well. Need further investigation

## Reference:
- https://en.rattibha.com/thread/1470406419786698761
- https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson
- https://kikaben.com/transformers-coding-details/
- https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
- https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
- on torchtext: https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=X_tml54u-6AS

## 1 SelfAttaention

self-attention in simple words is attention on the same sequence. I like to define it as a layer that tells you which 
token loves another token in the same sequence. for self-attention, the input is passed through 3 linear layers: query, 
key, value.

![transformer_pic](./img/Selfattention_forward0.PNG)


In the forward function, we apply the formula for self-attention. softmax(Q.K´/ dim(k))V. torch.bmm does matrix 
multiplication of batches. dim(k) is the sqrt of k. Please note: q, k, v (inputs) are the same in the case of 
self-attention.

Let's look at the forward function and the formula for self-attention (scaled). Ignoring the mask part, everything is 
pretty easy to implement.

![transformer_pic](./img/Selfattention_forward.PNG)


The mask just tells where not to look (e.g. padding tokens)

![transformer_pic](./img/Selfattention_mask.PNG)



## MultiHeadAttention
Now comes the fun part. Multi-head attention. We see it 3 times in the architecture. Multi-headed attention is nothing 
but many different self-attention layers. The outputs from these self-attentions are concatenated to form output the 
same shape as input.

![transformer_pic](./img/MultiHeadAttention.PNG)


## EncoderLayer
Let's take a look at the encoder layer. It consists of multi-headed attention, a feed forward network and two layer 
normalization layers. See forward(...) function to understand how skip-connection works. Its just adding original 
inputs to the outputs.

![transformer_pic](./img/EncoderLayer.PNG)

## Encoder
The encoder is composed of N encoder layers. Let's implement this as a black box too. The output of one encoder goes 
as input to the next encoder and so on. The source mask remains the same till the end.

![transformer_pic](./img/Encoder.PNG)


## DecoderLayer
The decoder layer consists of two different types of attention. the masked version has an extra mask in addition to 
padding mask. We will come to that. The normal multi-head attention takes key and value from final encoder output. key 
and value here are same.

![transformer_pic](./img/DecoderLayer1.PNG)

Query comes from output of masked multi-head attention (after layernorm). Checkout the forward function and things are 
very easy to understand.

![transformer_pic](./img/DecoderLayer2.PNG)

## Decoder
Similarly, we have the decoder composed of decoder layers. The decoder takes input from the last encoder layer and the 
target embeddings and target mask. enc_mask is the same as src_mask as explained previously.

![transformer_pic](./img/Decoder.PNG)

## Transformer
There are two parts: encoder and decoder. Encoder takes source embeddings and source mask as inputs and decoder takes 
target embeddings and target mask. Decoder inputs are shifted right. What does shifted right mean? 

![transformer_pic](./img/TransformerX.PNG)


## Overall Network
![transformer_pic](./img/OverallNetwork.PNG)






