image[[280, 86, 763, 303]]


figure_title[[171, 343, 824, 373]]
Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

text[[171, 398, 824, 427]]
of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

sub_title[[172, 440, 433, 455]]
## 3.2.1 Scaled Dot-Product Attention

text[[171, 464, 825, 520]]
We call our particular attention "Scaled Dot-Product Attention" (Figure \( ^{2} \) ). The input consists of queries and keys of dimension  \( d_{k} \) , and values of dimension  \( d_{v} \) . We compute the dot products of the query with all keys, divide each by  \( \sqrt{d_{k}} \) , and apply a softmax function to obtain the weights on the values.

text[[171, 526, 825, 569]]
In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:

equation[[355, 585, 823, 618]]
 \[ \mathrm{A t t e n t i o n}(Q,K,V)=\operatorname{s o f t m a x}(\frac{Q K^{T}}{\sqrt{d_{k}}})V \quad (1) \] 

text[[171, 627, 826, 715]]
The two most commonly used attention functions are additive attention  \( [2] \) , and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of  \( \frac{1}{\sqrt{d_{k}}} \) . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

text[[171, 720, 826, 779]]
While for small values of  \( d_{k} \)  the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of  \( d_{k}\left[\underline{3}\right] \) . We suspect that for large values of  \( d_{k}, \)  the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients \( ^{4} \) . To counteract this effect, we scale the dot products by  \( \frac{1}{\sqrt{d_{k}}} \) .

sub_title[[172, 793, 379, 808]]
## 3.2.2 Multi-Head Attention

text[[171, 817, 826, 874]]
Instead of performing a single attention function with  \( d_{model} \) -dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to  \( d_{k} \) ,  \( d_{k} \)  and  \( d_{v} \)  dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding  \( d_{v} \) -dimensional