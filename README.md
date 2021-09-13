## Pytorch Vector Quantization
A vector quantization library based on pytorch. 

Implementated methods:
- [x] Vector Quantization
- [x] Vector Quantization based on momentum moving average
- [x] Vector Quantization based on gumbel-softmax trick
- [x] Product Quantization
- [x] Residual Quantization

## Usage

```python
import torch
from vector_quantize import VectorQuantizer

vq = VectorQuantizer(
    n_e = 1024,          # codebook vocalbulary size
    e_dim = 256,         # codebook vocalbulary dimension
    beta = 1.0,          # the weight on the commitment loss
)

x = torch.randn(1, 256, 16, 16)          # size of NCHW
quantized, commit_loss, indices = vq(x)          # shape of (1, 256, 16, 16), (1), (1, 16, 16)
```

