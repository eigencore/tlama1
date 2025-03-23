import torch
import torch.nn.functional as F
import time

from tlama import Attention
from tlama import TlamaConfig

class SelfAttention(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = torch.nn.Linear(config.d_model, 3 * config.d_model)
        # output projection
        self.c_proj = torch.nn.Linear(config.d_model, config.d_model)
        self.c_proj.TLAMA124M_SCALE_INIT = 1
        # regularization
        self.n_heads = config.n_heads
        self.d_model = config.d_model

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

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
    m = torch.arange(end, device=theta_.device, dtype=torch.float32)
    freqs = torch.outer(m, theta_)  # m_i * theta_j
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # r*(cos(m_i * theta_j), sin(m_i * theta_j)), r=1
    return freqs_cis


# Crear configuración de prueba
config = TlamaConfig(
    d_model=768,
    n_heads=12,
    n_layers=12,
    use_parallel=False,
)


# Crear tensores de entrada simulados
x = torch.randn(config.max_batch_size, config.max_seq_len, config.d_model).cuda()
freq_cis = compute_freqs_cis(
                config.d_model // config.n_heads,  # head_dim
                config.max_seq_len , # need to multiply by 2?
            ).cuda()

# Instanciar modelos
attention1 = Attention(config).cuda()
attention2 = SelfAttention(config).cuda()

# Pruebas de rendimiento
def benchmark(model, name, iters=10):
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        torch.cuda.synchronize()  # Asegurar que la GPU ha terminado todo antes de medir
        starter.record()
        
        with torch.no_grad():
            output = model(x, freq_cis) if name == "Attention" else model(x)
        
        ender.record()
        torch.cuda.synchronize()  # Esperar a que termine todo antes de medir
        times.append(starter.elapsed_time(ender))  # Tiempo en milisegundos

    avg_time = sum(times) / iters
    print(f"{name}: {avg_time:.2f} ms/iter")

# Comparar rendimiento
benchmark(attention1, "Attention")
benchmark(attention2, "SelfAttention")

# Comparar uso de memoria
torch.cuda.empty_cache()
mem1 = torch.cuda.memory_allocated() / 1e6
benchmark(attention1, "Attention")
mem2 = torch.cuda.memory_allocated() / 1e6
print(f"Memoria usada - Attention: {mem1:.2f} MB")
print(f"Memoria usada - SelfAttention: {mem2:.2f} MB")

# Comparar calidad de salida
out1 = attention1(x, freq_cis)
out2 = attention2(x)
diff = torch.abs(out1 - out2).mean().item()
print(f"Diferencia media en salida: {diff:.6f}")
