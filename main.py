import torch, numpy as np, torchvision, utils
from torch import nn

if torch.cuda.is_available():
    torch.cuda.set_device(torch.cuda.device_count()-1)

batch_size = 32
n_types = 10

acgan = utils.ACGAN(batch_size)
#res = utils.ResNet18(n_types)

acgan.train()
