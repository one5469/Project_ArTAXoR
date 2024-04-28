import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms, v2
import os

baseTF = transforms.Compose(transforms=[
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
    # ,transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

