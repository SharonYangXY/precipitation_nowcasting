import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print (torch.__version__)
# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)


