#checking cuda and if quantization is working

import torch
import bitsandbytes as bnb

print(torch.cuda.is_available())
#print(bnb.cuda.is_available())

#print("loaded")