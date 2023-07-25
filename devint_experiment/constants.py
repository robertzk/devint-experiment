import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTEXT_WINDOW_LENGTH = 35
