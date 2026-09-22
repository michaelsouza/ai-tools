text[[171, 92, 824, 121]]
output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure \( ^{2} \) 

text[[172, 126, 824, 155]]
Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

equation[[300, 182, 696, 222]]
 \[ \begin{aligned}MultiHead(Q,K,V)&=Concat(head_{1},...,head_{h})W^{O}\\ where head_{i}&=Attention(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})\end{aligned} \] 

text[[171, 255, 822, 286]]
Where the projections are parameter matrices  \( W_{i}^{Q} \in R^{d_{\text{model}} \times d_{k}} \) ,  \( W_{i}^{K} \in R^{d_{\text{model}} \times d_{k}} \) .  \( W_{i}^{V} \in R^{d_{model} \times d_{v}} \)  and  \( W^{O} \in R^{hd_{v} \times d_{\text{model}}} \) .

text[[171, 293, 825, 336]]
In this work we employ h = 8 parallel attention layers, or heads. For each of these we use  \( d_{k} = d_{v} = d_{model}/h = 64 \) . Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

sub_title[[172, 351, 498, 366]]
## 3.2.3 Applications of Attention in our Model

text[[171, 375, 611, 390]]
The Transformer uses multi-head attention in three different ways:

text[[216, 402, 826, 471]]
• In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as  \( [38][2][9] \) .

text[[216, 477, 825, 532]]
- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.

text[[216, 538, 825, 609]]
- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to  \( -\infty \) ) all values in the input of the softmax which correspond to illegal connections. See Figure \( ^{2} \) 

sub_title[[172, 625, 481, 639]]
## 3.3 Position-wise Feed-Forward Networks

text[[171, 651, 825, 694]]
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

equation[[368, 714, 823, 729]]
 \[ \mathrm{FFN}(x)=\max(0,xW_{1}+b_{1})W_{2}+b_{2} \quad (2) \] 

text[[171, 742, 826, 799]]
While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is  \( d_{model} = 512 \) , and the inner-layer has dimensionality  \( d_{ff} = 2048 \) .

sub_title[[172, 815, 393, 830]]
## 3.4 Embeddings and Softmax

text[[171, 841, 827, 912]]
Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension  \( d_{model} \) . We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by  \( \sqrt{d_{model}} \) .