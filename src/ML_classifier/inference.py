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
import joblib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import *
from preprocessing import *
from simple_check import *
from training import *

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
'''
The following functions are used to generate predictions on a test set (X_test is a list of strings, and y_test is an array of 1 or 0 values.) 
1. Using a Heuristic (Non-ML model) which uses simple preprocessing.
2. Using an NLP based technique with Logistic Regression classifier.
3. Using a 1D-CNN to classify text sequence.
'''

#################################### Get predictions for Heuristic (Non-ML) Model ####################################


def get_pred_heuristics(X_test):
    
    preds_heuristics=[]
    for test_doc in list(X_test):
        test_doc = preprocess_text(test_doc)
        pred = check(test_doc)
        preds_heuristics.append(pred) 
    
    preds_heuristics = np.array(preds_heuristics, dtype=np.int64)
    
    print("Predictions stored.")
    return preds_heuristics


#################################### Get predictions for NLP Classifier ####################################


####### Load stored model #######

def load_NLP(path_to_model="models/NLP_model.pkl"):
    model = joblib.load(path_to_model)
    clf = model["clf"]
    tfidf_vec = model["tfidf"]
    scaler = model["scaler"]
    alphabet = model["alphabet"]
    return clf, scaler, tfidf_vec, alphabet


####### Testing (with windowing - if input is too long) #######


# def unicode_dna_preprocessor(text: str) -> str:
#     """
#     Normalizes text by stripping invisible characters and transliterating 
#     Unicode homoglyphs to ASCII to ensure clean genomic data.
#     """
#     # Standard Normalize (Fixes wide text 'Ａ' -> 'A')
#     text = unicodedata.normalize('NFKC', text)
    
#     # Remove Invisible Characters (Category Cf = Format, Cc = Control)
#     text = "".join(ch for ch in text if unicodedata.category(ch) not in ["Cf", "Cc"])
    
#     # Maps unicode to nearest ASCII
#     text = unidecode(text)
#     text = text.upper()

#     return text


# def preprocess_text(text):
#     """
#     Very minimal preprocessing:
#     - Lowercase
#     - Remove whitespace newlines
#     We DO NOT filter characters, because DNA may be obfuscated.
#     - Unicode detection processing
#     """
#     text = text.lower().replace("\n", " ").replace("\t", " ")
#     #text = unicode_dna_preprocessor(text)
#     return text


# def get_kmers(text, k=3):
#     """Extract k-mers without filtering characters."""
#     kmers = []
#     for i in range(len(text) - k + 1):
#         kmers.append(text[i:i+k])
#     return kmers


# ######### Markov Transition Features #########


# def markov_features(text, alphabet):
#     """
#     Build a Markov transition probability vector from:
#     P(next_char | current_char)

#     The output is flattened row-major into a single vector.
#     """
#     if len(text) < 2:
#         return np.zeros(len(alphabet) * len(alphabet))

#     index = {c: i for i, c in enumerate(alphabet)}

#     # Transition counts
#     trans = np.zeros((len(alphabet), len(alphabet)))

#     for a, b in zip(text[:-1], text[1:]):
#         if a in index and b in index:
#             trans[index[a], index[b]] += 1

#     row_sums = trans.sum(axis=1, keepdims=True)
#     row_sums[row_sums == 0] = 1  
#     trans = trans / row_sums

#     return trans.flatten()


# ######### Single-document Feature Extraction #########


# def extract_features_single(
#     text,
#     k,
#     tfidf_vectorizer,
#     alphabet,
#     fit_tfidf=False
# ):
#     """
#     Extract TF-IDF + entropy + Markov vector.
#     All vectors are guaranteed fixed length if:
#     - TF-IDF was fit on the full training corpus
#     - alphabet is global
#     """
#     pre = preprocess_text(text)
#     kmers = get_kmers(pre, k)

#     entropy = shannon_entropy(pre)
#     markov_vec = markov_features(pre, alphabet)

#     kmers_str = " ".join(kmers)

#     if fit_tfidf:
#         tfidf_vec = tfidf_vectorizer.fit_transform([kmers_str]).toarray()[0]
#     else:
#         tfidf_vec = tfidf_vectorizer.transform([kmers_str]).toarray()[0]

#     return np.concatenate([tfidf_vec, [entropy], markov_vec])

def classify_document(
    text,
    clf,
    scaler,
    tfidf_vec,
    alphabet,
    k=3,
    max_window_chars=20000
):
    """
    If document is short -> classify whole.
    If too long -> split into windows.
    If ANY window predicts 1 -> output 1 (DNA found).
    """
    if len(text) <= max_window_chars:
        # No windowing
        feats = extract_features_single(
            text, k, tfidf_vec, alphabet, fit_tfidf=False
        )
        feats = scaler.transform([feats])
        pred = clf.predict(feats)[0]
        return pred

    # WINDOWING
    windows = [
        text[i:i + max_window_chars]
        for i in range(0, len(text), max_window_chars)
    ]

    for w in windows:
        feats = extract_features_single(
            w, k, tfidf_vec, alphabet, fit_tfidf=False
        )
        feats = scaler.transform([feats])
        pred = clf.predict(feats)[0]
        if pred == 1:
            return 1

    return 0  # none of the windows detected DNA

def get_pred_NLP(X_test, clf, scaler, tfidf_vec, alphabet):
    preds=[]

    #print("length of alphabet:", len(alphabet))
    
    for test_doc in X_test:
        # print(repr(test_doc))
        test_doc = preprocess_text(test_doc)
        # print(repr(test_doc))
        pred = classify_document(
        test_doc,
        clf,
        scaler,
        tfidf_vec,
        alphabet,
        k=3)
        preds.append(pred)
    
    preds = np.array(preds, dtype=np.int64)
    #print("Predictions stored.")

    return preds


    
#################################### Get predictions for CNN Model ####################################


####### Load stored model #######

def load_CNN(path_to_model="models/CNN_model.pth"):
    # char_encoder = CharEncoder() 
    char_encoder_path = os.path.join(project_root, "models", "char_encoder.pkl")
    char_encoder = joblib.load(char_encoder_path)
    # print("Vocab size:", char_encoder.vocab_size)
    model = DNASequenceCNN(
        vocab_size=char_encoder.vocab_size,
        embed_dim=32,
        num_classes=2
    ).to(device)
    
    #model.load_state_dict(torch.load(path_to_model))
    state_dict = torch.load(path_to_model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()   # set to inference mode
    return model, char_encoder

####### Get predictions on Test set #######

def get_pred_CNN(X_test, model, char_encoder):
    preds_cnn = []
    probs_cnn = []
    max_len=28781 # from training corpora
    for test_doc in list(X_test):
        with torch.no_grad():
            test_doc_proc = preprocess_text(test_doc)
    
            # Transform + pad
            test_seq = char_encoder.transform(test_doc_proc) 
            test_seq = np.pad(test_seq, (0, max(0, max_len - len(test_seq))),
                              constant_values=0)
            #test_seq = torch.tensor([test_seq], dtype=torch.long).to(device)
            test_seq = np.array(test_seq, dtype=np.int64)     # convert list to single numpy array
            test_seq = torch.from_numpy(test_seq).long().unsqueeze(0).to(device)
    
            test_entropy = torch.tensor([[shannon_entropy(test_doc_proc)]],
                                        dtype=torch.float).to(device)
    
            pred = model(test_seq, test_entropy)
            probs = F.softmax(pred, dim=1)
    
        preds_cnn.append(torch.argmax(pred, dim=1).item())
        probs_cnn.append(probs.cpu())  # move back to CPU if needed

    return preds_cnn


#################################### Get performance metrics ####################################


# Print overall performance of the test data
def get_performance_metrics(preds,y_test):
    y_pred = preds
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    return acc,prec,rec,f1

    
# Print performance metrics for each adversarial type in the test data
def print_performance_for_adversarial_types(y_pred, y_test, type_test):
    for i in list(set(type_test)):
        print("Adversarial Attempt Type:", adversarial_attempt_type[i])
        y_pred_temp = y_pred[type_test==i]
        y_test_temp = y_test[type_test==i]
        acc  = accuracy_score(y_test_temp, y_pred_temp)
        prec = precision_score(y_test_temp, y_pred_temp)
        rec  = recall_score(y_test_temp, y_pred_temp)
        f1   = f1_score(y_test_temp, y_pred_temp)
        print("Accuracy :", acc)
        print("Precision:", prec)
        print("Recall   :", rec)
        print("F1 Score :", f1)


#################################### Get predictions on Unseen Adversarial Dataset ####################################

# generates predictions based on 
def generate_predictions(input_list, model_type='nlp', model_path='models/CNN_model.pth'):
    
    if (model_type=='nlp'):
        # Load pretrained weights
        model_path = os.path.join(project_root, model_path)
        clf, scaler, tfidf_vec, alphabet = load_NLP(model_path)
        # generate predictions
        preds = get_pred_NLP(input_list, clf, scaler, tfidf_vec, alphabet)
    elif (model_type=='cnn'):
        # Load pretrained weights
        model_path = os.path.join(project_root, model_path)
        model, char_encoder = load_CNN(model_path)
        # generate predictions
        preds = get_pred_CNN(input_list, model, char_encoder)
    else: #heuristics
        preds = get_pred_heuristics(input_list)
    return preds


def get_perf_unseen(path_to_data, model_type = 'nlp'):
    adv = pd.read_pickle(path_to_data)
    sequences = list(adv['Adversarial sequence'].values)
    types = list(set(adv["type"].values))
    for t in types:
        print("Adversarial Type:",t)
        df = adv[adv['type']==t]
        seq = list(df['Adversarial sequence'].values)
        preds = generate_predictions(seq, model_type)
        print("Accuracy:",(preds.sum()/len(preds)))
    return preds