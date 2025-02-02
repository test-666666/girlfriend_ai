import torch
import bitsandbytes as bnb
import bitsandbytes
hasattr(bitsandbytes, 'configuration')
print(bitsandbytes.__version__)
print(bnb.__version__)
print(torch.cuda.is_available())