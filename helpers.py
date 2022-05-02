import os
import numpy as np
import torch
import random

from settings import path

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log(line, file): #####################
  with open(path + file, 'a+') as log:
      content = log.read()
      log.write(content + line + str("\n"))

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def set_seed(seed):
    torch.manual_seed(seed)                  # pytorch
    random.seed(seed)                        # python
    np.random.seed(seed)                     # numpy
    torch.use_deterministic_algorithms(True) # CNN
    g = torch.Generator()                    # dataloaders
    return g.manual_seed(0)
