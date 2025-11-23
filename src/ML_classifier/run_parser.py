import os
from typing import Union, Tuple, Optional, Set
from pypdf import PdfReader
from docx import Document
import pandas as pd
import numpy as np
import string
import random
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
random.seed = 42
import pickle as pkl
import unicodedata
from unidecode import unidecode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import *
from preprocessing import *
from prep_training_data import *
from heuristics.simple_check import *
from training import *
from inference import *



takes an input file 


displays whether it has genomic sequence or not



analyse file type -> if true, flag, else parse content


preprocess content


use NLP model (can change setting to run either heuristics or CNN)

# load model
# display result

 