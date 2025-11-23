import os
from typing import Union, Tuple, Optional, Set
from pypdf import PdfReader
from docx import Document
import torch
import pandas as pd
import numpy as np
import string
import random
from sklearn.model_selection import train_test_split
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
random.seed = 42
import pickle as pkl


### Split dataset into train and test using 80-20 split ###
def split_train_test(df):
    # Compute split index - train and test
    split_idx = int(len(df) * 0.80)  # first 80%
    
    # First 80% and remaining 20%
    train = df[:split_idx]
    test = df[split_idx:]
    return train,test

### Shuffle a given list ###
def shuffle_lists(list_of_lists):
    out = []
    for lst in list_of_lists:
        lst_copy = lst[:]            # avoid modifying original
        random.shuffle(lst_copy)     # shuffle in place
        out.append(lst_copy)
    return out


### Generate random string of text (used to generate negative samples) ###
CHARS = string.ascii_letters + string.digits + " .,!?$%&*#@/\n"

def random_garbage(n):
    return ''.join(random.choices(CHARS, k=n))

def generate_fake_content():
    prefix_len = random.randint(0, 5000)
    suffix_len = random.randint(0, 5000)

    prefix = random_garbage(prefix_len)
    suffix = random_garbage(suffix_len)

    return f"{prefix}{suffix}"

