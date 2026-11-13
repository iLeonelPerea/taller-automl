"""Sanidad de PyTorch: un paso de entrenamiento baja la pérdida y CUDA calcula bien.

torch es una dependencia fija del taller (ver pyproject.toml): estas dos
pruebas comprueban que el paquete instalado funciona; que el taller entrena
en la tarjeta de video lo prueba `tests/test_juguete.py`. Las dos entrenan en
la tarjeta de video: llevan la marca `gpu`, que `tests/conftest.py` hace
fallar con un mensaje claro cuando no hay una tarjeta de video NVIDIA con
CUDA visible, en vez de saltarse.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_torch_entrena_un_paso_en_la_tarjeta_de_video():
    # Generador propio con semilla explícita, sin tocar el generador aleatorio global de torch
    # (convención del repo): skip_init evita que Linear sortee sus pesos con él.
    g = torch.Generator(device="cuda").manual_seed(0)
    capa = torch.nn.utils.skip_init(torch.nn.Linear, 4, 1, device="cuda")
    with torch.no_grad():
        capa.weight.copy_(torch.randn(1, 4, generator=g, device="cuda"))
        capa.bias.copy_(torch.randn(1, generator=g, device="cuda"))
    optimizador = torch.optim.SGD(capa.parameters(), lr=0.1)
    x = torch.randn(8, 4, generator=g, device="cuda")
    y = torch.randn(8, 1, generator=g, device="cuda")

    antes = torch.nn.functional.mse_loss(capa(x), y).item()
    perdida = torch.nn.functional.mse_loss(capa(x), y)
    perdida.backward()
    optimizador.step()
    despues = torch.nn.functional.mse_loss(capa(x), y).item()

    assert despues < antes


def test_gpu_cuda_calcula():
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(256, 256, device="cuda", generator=g)
    b = a @ a.T
    torch.cuda.synchronize()
    assert b.device.type == "cuda"
    assert torch.allclose(b, b.T, atol=1e-3)
