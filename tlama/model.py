# Copyright 2025 EigenCore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TlamaConfig:
    """
    Configuration class for the Tlama model.
    
    Defines model hyperparameters such as the number of layers, dimensions, and other key settings.
    
    Attributes:
        d_model (int): Dimensionality of the model. Default is 4096.
        n_layers (int): Number of transformer layers. Default is 32.
        n_kv_heads (Optional[int]): Number of key-value heads for attention. Default is None (follows n_heads).
        vocab_size (int): Size of the vocabulary. Default is -1 (must be set manually).
        multiple_of (int): Ensures the hidden layer size in SwiGLU is a multiple of this value. Default is 256.
        ffn_dim_multiplier (Optional[float]): Multiplier for feed-forward network dimension. Default is None.
        norm_eps (float): Epsilon value for normalization layers. Default is 1e-5.
        rope_theta (float): Theta value for RoPE positional embeddings. Default is 500000.
        max_batch_size (int): Maximum batch size. Default is 32.
        max_seq_len (int): Maximum sequence length. Default is 2048.
    """
    d_model: int = 4096
    n_layers: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # Ensures hidden layer size in SwiGLU is a multiple of this value
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    
    max_batch_size: int = 32
    max_seq_len: int = 2048

    


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    This normalization is based on the Root Mean Square (RMS) norm instead of mean and variance,
    as in LayerNorm. It is commonly used in language models to improve stability and efficiency.
    
    Attributes:
        eps (float): Small value to prevent division by zero. Default is 1e-6.
        weight (torch.nn.Parameter): Learnable parameter to scale the output.
    
    Parameters:
        dim (int): Input dimension over which normalization is applied.
        eps (float, optional): Value for numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).
        
        Returns:
            torch.Tensor: Normalized and scaled tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Computes the complex-valued rotary positional embeddings (RoPE) frequencies.

    This function generates a tensor of complex numbers representing the rotary positional embeddings
    used in transformer models. The embeddings are computed in polar form, where the magnitude is 1
    and the angle is determined by the product of position indices and frequency scaling factors.

    Args:
        dim (int): Dimensionality of the embeddings. Typically corresponds to the model's hidden size.
        end (int): Maximum sequence length for which the embeddings are computed.
        theta (float, optional): Scaling factor for the frequencies. Default is 10000.0.

    Returns:
        torch.Tensor: A tensor of shape `(end, dim // 2)` containing complex numbers in polar form.
                      Each complex number has a magnitude of 1 and an angle determined by the
                      position and frequency.
    """
    theta_ = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    m = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(m, theta_)  # m_i * theta_j
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # r*(cos(m_i * theta_j), sin(m_i * theta_j)), r=1
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes the `freqs_cis` tensor for broadcasting with the input tensor `x`.

    This function adjusts the shape of the `freqs_cis` tensor so that it can be broadcasted
    with the input tensor `x` during operations such as element-wise multiplication. The reshaped
    tensor will have singleton dimensions (`1`) for all axes except the sequence length and embedding
    dimensions, ensuring compatibility with `x`.

    Args:
        freqs_cis (torch.Tensor): A tensor of shape `(seq_len, hidden_dim)` containing complex-valued
                                  rotary positional embeddings.
        x (torch.Tensor): Input tensor of shape `(batch_size, seq_len, hidden_dim)` or similar, where
                          `seq_len` is the sequence length and `hidden_dim` is the embedding size.

    Returns:
        torch.Tensor: A reshaped version of `freqs_cis` with singleton dimensions added, making it
                      compatible for broadcasting with `x`.

    Example:
        >>> freqs_cis = torch.randn(2048, 256)  # (seq_len, hidden_dim)
        >>> x = torch.randn(8, 2048, 256)      # (batch_size, seq_len, hidden_dim)
        >>> reshaped_freqs_cis = reshape_for_broadcast(freqs_cis, x)
        >>> print(reshaped_freqs_cis.shape)
        torch.Size([1, 2048, 256])
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, "Input tensor `x` must have at least 2 dimensions."
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        "Shape of `freqs_cis` must match the sequence length and embedding size of `x`."
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embeddings (RoPE) to the query (`xq`) and key (`xk`) tensors.

    This function incorporates positional information into the query and key tensors by applying
    rotary positional embeddings. The embeddings are applied in the complex domain, where the
    tensors are first converted to complex numbers, multiplied by the positional embeddings, and
    then converted back to real numbers.

    Args:
        xq (torch.Tensor): Query tensor of shape `(batch_size, seq_len, hidden_dim)` containing real values.
        xk (torch.Tensor): Key tensor of shape `(batch_size, seq_len, hidden_dim)` containing real values.
        freqs_cis (torch.Tensor): Complex-valued rotary positional embeddings of shape `(seq_len, hidden_dim // 2)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - `xq_out` (torch.Tensor): Query tensor with positional embeddings applied, of shape `(batch_size, seq_len, hidden_dim)`.
            - `xk_out` (torch.Tensor): Key tensor with positional embeddings applied, of shape `(batch_size, seq_len, hidden_dim)`.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


