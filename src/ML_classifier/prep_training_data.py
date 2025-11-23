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
random.seed = 42
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import *
from preprocessing import *
from heuristics.simple_check import *
from training import *
from inference import *

data_root_path = "data/raw/" # file path with raw training data
path_to_save_train_data = "" # file path where we will save the final training data
path_to_save_test_data = "" # file path where we will save the final test data

""" 
### Curating our dataset -> Methodology

To curate our training and test data, we generate the following type of adversarial cases from raw DNA sequences. Below are the different adversarial types of data used, along with the number of positive and negative samples generated for each type. For more details about the adversarial type, please check the ReadMe or report. 

1) english and seq -> pos(actual), pos(actual+english), neg(english)
                   200          200                  200          

2) text and seq -> pos(actual), pos(actual+text) 
                1000         1000               

3) text_and_seq_alternate_random_characters -> pos(actual), pos(actual with alt rand char+english), neg(english)
                                            200          200                                     200            

4) text and seq compressed -> pos(actual), pos(actual compressed), pos(actual compressed + english), neg(english)
                           200          200                     200                               200           

5) text and seq compressed and replaced -> pos (actual), pos (actual compressed), pos (compressed_replaced + english), neg (english)
                                        200           200                      200                                  200              

6) text and seq replaced with other characters -> pos(actual), pos(replaced + english), neg(english)
                                            200             200                      200            

7) single (fake) dna -> neg
                        100  


8) multicharacter mapping -> pos (actual), pos (multicharacter + english), neg (english)
                          200           200                             200                

9) sequence with random space and english text -> pos(actual), pos(actual with random space), pos (actual with random space + english), neg (english)
                                                200         200                            200                                       200


10) sequence with with text breaks -> pos(actual), pos(actual with english text breaks), neg(english)
                                  200          200                                   200

11) randomly generated text -> neg
                               4100  (calculated based on above imbalance of pos and neg)


Total input data samples = 11600
Total pos = 5800 (50%)
Total neg = 5800 (50%)
Train = 80%, Test = 20%

Distributions are same across train and test set.
"""

#################################### Create map of different adversarial types ####################################

adversarial_attempt_type = {1:"Sequence (continuous) with English Text", 
                            2:"Sequence (continuous) with Random Text (All positive samples)", 
                            3:"Sequence with alternate random characters, bounded by English Text",
                            4:"Sequence compressed and obfuscated with English Text",
                            5:"Sequence compressed and replaced by other characters",
                            6:"Sequence replaced by other characters",
                            7:"Fake DNA sequences (All negative samples)",
                            8:"Sequence replaced with multi-character mapping(length:1-5) and bounded with English Text",
                            9:"Sequence broken with random spaces, bounded by English Text",
                            10:"Sequence broken by random strings of text (variable length)",
                            11:"Randomly generated characters (All negative samples)"}


#################################### Generate adversarial cases to train and test our model from raw DNA sequences ####################################

pos_train=[]
neg_train=[]

pos_test=[]
pos_test_type=[] # to store adversarial attack type

neg_test=[]
neg_test_type=[] # to store adversarial attack type


p1=[]
p2=[]
n1=[]

######################### TYPE 1 #########################


path = "english_and_seq"
for file_name in os.listdir(path):
    file_path = path + "/" + file_name
    if (not (file_name.endswith(".csv") | file_name.endswith(".txt"))):
        continue
    df = pd.read_csv(file_path)
    p1.append(df['sequence'].values[0])
    p2.append(df['text_with_dna'].values[0])
    n1.append(df['text_without_dna'].values[0])

p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
n1_train, n1_test = split_train_test(n1)
# add to global lists
pos_train = pos_train + p1_train + p2_train
pos_test = pos_test + p1_test + p2_test
pos_test_type = pos_test_type + [1]*len(p1_test + p2_test)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [1]*len(n1_test)


######################### TYPE 2 #########################

p1=[]
p2=[]

path = "text_and_seq"
for file_name in os.listdir(path):
    file_path = path + "/" + file_name
    if (not (file_name.endswith(".csv") | file_name.endswith(".txt"))):
        continue
    df = pd.read_csv(file_path)
    p1.append(df['sequence'].values[0])
    p2.append(df['generated'].values[0])

p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
# add to global lists
pos_train = pos_train + p1_train + p2_train
pos_test = pos_test + p1_test + p2_test
pos_test_type = pos_test_type + [2]*len(p1_test + p2_test)

######################### TYPE 3 #########################

p1=[]
p2=[]
n1=[]

path = "text_and_seq_alternate_random_characters"
for file_name in os.listdir(path):
    file_path = path + "/" + file_name
    if (not (file_name.endswith(".csv") | file_name.endswith(".txt"))):
        continue
    df = pd.read_csv(file_path)
    p1.append(df['sequence'].values[0])
    p2.append(df['text_with_dna'].values[0])
    n1.append(df['text_without_dna'].values[0])

p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
n1_train, n1_test = split_train_test(n1)
# add to global lists
pos_train = pos_train + p1_train + p2_train
pos_test = pos_test + p1_test + p2_test
pos_test_type = pos_test_type + [3]*len(p1_test + p2_test)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [3]*len(n1_test)


######################### TYPE 4 #########################


p1=[]
p2=[]
p3=[]
n1=[]

path = "text_and_seq_compressed"
for file_name in os.listdir(path):
    file_path = path + "/" + file_name
    if (not (file_name.endswith(".csv") | file_name.endswith(".txt"))):
        continue
    df = pd.read_csv(file_path)
    p1.append(df['sequence'].values[0])
    p2.append(df['compressed'].values[0])
    p3.append(df['text_with_dna'].values[0])
    n1.append(df['text_without_dna'].values[0])

p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
p3_train, p3_test = split_train_test(p3)
n1_train, n1_test = split_train_test(n1)
# add to global lists
pos_train = pos_train + p1_train + p2_train + p3_train
pos_test = pos_test + p1_test + p2_test + p3_test
pos_test_type = pos_test_type + [4]*len(p1_test + p2_test + p3_test)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [4]*len(n1_test)



######################### TYPE 5 #########################


p1=[]
p2=[]
p3=[]
n1=[]

path = "text_and_seq_compressed_and_replaced"
for file_name in os.listdir(path):
    file_path = path + "/" + file_name
    if (not (file_name.endswith(".csv") | file_name.endswith(".txt"))):
        continue
    df = pd.read_csv(file_path)
    p1.append(df['sequence'].values[0])
    p2.append(df['compressed'].values[0])
    p3.append(df['text_with_dna'].values[0])
    n1.append(df['text_without_dna'].values[0])

p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
p3_train, p3_test = split_train_test(p3)
n1_train, n1_test = split_train_test(n1)
# add to global lists
pos_train = pos_train + p1_train + p2_train + p3_train
pos_test = pos_test + p1_test + p2_test + p3_test
pos_test_type = pos_test_type + [5]*len(p1_test + p2_test + p3_test)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [5]*len(n1_test)



######################### TYPE 6 #########################


p1=[]
p2=[]
n1=[]

path = "text_and_seq_replaced_with_other_characters"
for file_name in os.listdir(path):
    file_path = path + "/" + file_name
    if (not (file_name.endswith(".csv") | file_name.endswith(".txt"))):
        continue
    df = pd.read_csv(file_path)
    p1.append(df['sequence'].values[0])
    p2.append(df['text_with_dna'].values[0])
    n1.append(df['text_without_dna'].values[0])

p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
n1_train, n1_test = split_train_test(n1)
# add to global lists
pos_train = pos_train + p1_train + p2_train
pos_test = pos_test + p1_test + p2_test
pos_test_type = pos_test_type + [6]*len(p1_test + p2_test)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [6]*len(n1_test)


######################### TYPE 7 #########################


n1 = []
for file in os.listdir('single_dna'):
    try:
        with open('single_dna/'+file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except:
        continue
    n1.append(text)
n1 = [s for s in n1 if len(s) <= 1000]
n1 = random.sample(n1, 100)
n1_train, n1_test = split_train_test(n1)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [7]*len(n1_test)


######################### TYPE 8 #########################


df = pd.read_pickle('multicharacter_mapping.pkl') #sequence, text_with_dna, text_without_dna
p1 = list(df['sequence'].values)
p2 = list(df['text_with_dna'].values)
n1 = list(df['text_without_dna'].values)
p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
n1_train, n1_test = split_train_test(n1)
# add to global lists
pos_train = pos_train + p1_train + p2_train
pos_test = pos_test + p1_test + p2_test
pos_test_type = pos_test_type + [8]*len(p1_test + p2_test)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [8]*len(n1_test)



######################### TYPE 9 #########################

df = pd.read_pickle('sequence_with_random_space_and_english_text.pkl') #sequence, text_with_dna, text_without_dna
p1 = list(df['sequence'].values)
p2 = list(df['text_with_dna'].values)
p3 = list(df['sequence_with_random_space'].values)
n1 = list(df['text_without_dna'].values)
p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
p3_train, p3_test = split_train_test(p3)
n1_train, n1_test = split_train_test(n1)
# add to global lists
pos_train = pos_train + p1_train + p2_train + p3_train
pos_test = pos_test + p1_test + p2_test + p3_test
pos_test_type = pos_test_type + [9]*len(p1_test + p2_test + p3_test)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [9]*len(n1_test)


######################### TYPE 10 #########################


df = pd.read_pickle('sequence_with_text_breaks.pkl') #sequence, text_with_dna, text_without_dna
p1 = list(df['sequence'].values)
p2 = list(df['text_with_dna'].values)
n1 = list(df['text_without_dna'].values)
p1_train, p1_test = split_train_test(p1)
p2_train, p2_test = split_train_test(p2)
n1_train, n1_test = split_train_test(n1)
# add to global lists
pos_train = pos_train + p1_train + p2_train
pos_test = pos_test + p1_test + p2_test
pos_test_type = pos_test_type + [10]*len(p1_test + p2_test)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [10]*len(n1_test)

######################### TYPE 11 #########################

n1 = [] 

for i in range(4100):
    n1.append(generate_fake_content())

n1_train, n1_test = split_train_test(n1)
neg_train = neg_train + n1_train
neg_test = neg_test + n1_test
neg_test_type = neg_test_type + [11]*len(n1_test)


#################################### Build the training and test set ####################################

train_pos = pos_train
test_pos = pos_test
train_neg = neg_train
test_neg = neg_test

######################### Create and save training set #########################

# create pos dataframe
df_train_pos = pd.DataFrame(train_pos, columns=["data"])
df_train_pos["label"] = 1

# create neg dataframe
df_train_neg = pd.DataFrame(train_neg, columns=["data"])
df_train_neg["label"] = 0

# combine
df_combined = pd.concat([df_train_pos, df_train_neg], ignore_index=True)

#shuffle
df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
# Save dataframe
df_shuffled.to_pickle(path_to_save_train_data)

X = df_shuffled["data"].values
y = df_shuffled["label"].values

X_train, X_val, y_train, y_val = train_test_split(X,y,
                     test_size=0.1,
                     random_state=42,
                     stratify=y)

######################### Create and save test set #########################

# create pos dataframe
df_test_pos = pd.DataFrame({"data": test_pos, "type": pos_test_type})
df_test_pos["label"] = 1

# create neg dataframe
df_test_neg = pd.DataFrame({"data": test_neg, "type": neg_test_type})
df_test_neg["label"] = 0

# combine
df_combined = pd.concat([df_test_pos, df_test_neg], ignore_index=True)
# Save dataframe
df_combined.to_pickle(path_to_save_test_data)

X_test = df_combined["data"].values
y_test = df_combined["label"].values
type_test = df_combined["type"].values