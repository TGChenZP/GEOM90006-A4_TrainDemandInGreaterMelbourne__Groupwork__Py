import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
import pickle
import os
import copy

from tqdm import tqdm
import warnings

from sklearn.metrics import \
    r2_score, \
    mean_squared_error, \
    mean_absolute_error
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import torch.nn.functional as F

from typing import Tuple
from collections import defaultdict as dd

import gc