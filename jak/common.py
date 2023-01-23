import torch
import numpy as np

seed_all = lambda s: (np.random.seed(s), torch.manual_seed(s))
