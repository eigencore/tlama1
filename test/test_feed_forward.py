import torch
import torch.nn.functional as F
import time

from tlama import TlamaConfig
from tlama import FeedForward

class MLP(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = torch.nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu    = torch.nn.GELU(approximate='tanh')
        self.c_proj  = torch.nn.Linear(4 * config.d_model, config.d_model)
        self.c_proj.TLAMA124M_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# Crear configuración de prueba
config = TlamaConfig(
    d_model=768,
    n_heads=12,
    n_layers=12,
    use_parallel=False,
)


# Crear tensores de entrada simulados
x = torch.randn(config.max_batch_size, config.max_seq_len, config.d_model).cuda()

# Instanciar modelos
ffn1 = FeedForward(d_model=config.d_model, hidden_dim=4*config.d_model, multiple_of=config.multiple_of, ffn_dim_multiplier=config.ffn_dim_multiplier).cuda()
ffn2 = MLP(config).cuda()

# Pruebas de rendimiento
def benchmark(model, name, iters=10):
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        torch.cuda.synchronize()  # Asegurar que la GPU ha terminado todo antes de medir
        starter.record()
        
        with torch.no_grad():
            output = model(x)
        
        ender.record()
        torch.cuda.synchronize()  # Esperar a que termine todo antes de medir
        times.append(starter.elapsed_time(ender))  # Tiempo en milisegundos

    avg_time = sum(times) / iters
    print(f"{name}: {avg_time:.2f} ms/iter")

# Comparar rendimiento
benchmark(ffn1, "FeedForward")
benchmark(ffn2, "MLP")

# Comparar uso de memoria
torch.cuda.empty_cache()
mem1 = torch.cuda.memory_allocated() / 1e6
benchmark(ffn1, "Attention")
mem2 = torch.cuda.memory_allocated() / 1e6
print(f"Memoria usada - Attention: {mem1:.2f} MB")
print(f"Memoria usada - SelfAttention: {mem2:.2f} MB")

# Comparar calidad de salida
out1 = ffn1(x)
out2 = ffn2(x)
diff = torch.abs(out1 - out2).mean().item()
print(f"Diferencia media en salida: {diff:.6f}")
