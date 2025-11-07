import math
import torch
import torch.cuda.nvtx as nvtx

from einops import einsum, rearrange
from torch import Tensor
from jaxtyping import Bool, Float

from cs336_basics.nn_utils import softmax


@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("qk"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        torch.cuda.synchronize()

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
        torch.cuda.synchronize()

    with nvtx.range("av"):
        result = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
        torch.cuda.synchronize()
    return result