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
import joblib
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import *
from preprocessing import *
from simple_check import *
from inference import *


#################################### Load training and test data ####################################

# Read training and test data from saved csv/pkl
def get_train_test_values(train_path="df_train_final.pkl", test_path="df_test_final.pkl"):

    df_train = pd.read_pickle(train_path)
    df_test = pd.read_pickle(test_path)
    
    X = df_train["data"].values
    y = df_train["label"].values
    # Divide into training and validation (90-10 split)
    X_train, X_val, y_train, y_val = train_test_split(X,y,
                         test_size=0.1,
                         random_state=42,
                         stratify=y)
    
    X_test = df_test["data"].values
    y_test = df_test["label"].values
    type_test = df_test["type"].values

    return X_train, X_val, X_test, y_train, y_val, y_test, type_test


#################################### 1. Training NLP Model ####################################


######### K-mer Extraction #########


def get_kmers(text, k=3):
    """Extract k-mers without filtering characters."""
    kmers = []
    for i in range(len(text) - k + 1):
        kmers.append(text[i:i+k])
    return kmers


######### Markov Transition Features #########


def markov_features(text, alphabet):
    """
    Build a Markov transition probability vector from:
    P(next_char | current_char)

    The output is flattened row-major into a single vector.
    """
    if len(text) < 2:
        return np.zeros(len(alphabet) * len(alphabet))

    index = {c: i for i, c in enumerate(alphabet)}

    # Transition counts
    trans = np.zeros((len(alphabet), len(alphabet)))

    for a, b in zip(text[:-1], text[1:]):
        if a in index and b in index:
            trans[index[a], index[b]] += 1

    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  
    trans = trans / row_sums

    return trans.flatten()


######### Single-document Feature Extraction #########


def extract_features_single(
    text,
    k,
    tfidf_vectorizer,
    alphabet,
    fit_tfidf=False
):
    """
    Extract TF-IDF + entropy + Markov vector.
    All vectors are guaranteed fixed length if:
    - TF-IDF was fit on the full training corpus
    - alphabet is global
    """
    pre = preprocess_text(text)
    kmers = get_kmers(pre, k)

    entropy = shannon_entropy(pre)
    markov_vec = markov_features(pre, alphabet)

    kmers_str = " ".join(kmers)

    if fit_tfidf:
        tfidf_vec = tfidf_vectorizer.fit_transform([kmers_str]).toarray()[0]
    else:
        tfidf_vec = tfidf_vectorizer.transform([kmers_str]).toarray()[0]

    return np.concatenate([tfidf_vec, [entropy], markov_vec])



######### TRAINING PIPELINE #########


def train_pipeline_whole_docs(texts, labels, k=3):
    """
    Train using full documents (no windowing).
    Ensures fixed feature vector sizes for all samples.
    """
    processed_docs = [preprocess_text(t) for t in texts]

    # Build global alphabet across all training docs
    global_alphabet = sorted(list(set("".join(processed_docs))))

    # Build kmers for TF-IDF fitting
    kmers_docs = [" ".join(get_kmers(doc, k)) for doc in processed_docs]

    # Fit a single TF-IDF over ALL training documents
    tfidf_vec = TfidfVectorizer(analyzer="word")
    tfidf_vec.fit(kmers_docs)

    # Extract features
    feature_list = []
    for doc in processed_docs:
        f = extract_features_single(
            doc,
            k,
            tfidf_vec,
            global_alphabet,
            fit_tfidf=False
        )
        feature_list.append(f)

    X = np.vstack(feature_list)
    y = np.array(labels)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train classifier
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000
    )
    clf.fit(X_scaled, y)

    return clf, scaler, tfidf_vec, global_alphabet



def train_NLP(clf, scaler, tfidf_vec, alphabet, X_train, y_train, save_weights=False):

    print("Start training...")
    clf, scaler, tfidf_vec, alphabet = train_pipeline_whole_docs(X_train, y_train, k=3)
    print("Training done.")

    if(save_weights):
        NLP_model = {
        "clf": clf,
        "tfidf": tfidf_vec,
        "scaler": scaler,
        "alphabet": alphabet
        }
        joblib.dump(NLP_model, "models/NLP_model.pkl")

    return clf, scaler, tfidf_vec, alphabet


    
#################################### 2. Training CNN Model ####################################



######## Character Encoder #########


class CharEncoder:
    """
    Maps characters to integer IDs.
    0 is reserved for padding.
    Unknown characters map to 0 at test time.
    """
    def __init__(self, char2id={}, fitted= False, vocab_size=0):
        self.char2id = char2id
        self.id2char = {}
        self.fitted = fitted
        self.vocab_size = vocab_size

    def fit(self, texts):
        chars = sorted(list(set("".join(texts))))
        # index 0 = padding
        self.char2id = {c:i+1 for i,c in enumerate(chars)}
        self.id2char = {i+1:c for i,c in enumerate(chars)}
        self.vocab_size = len(self.char2id) + 1  # +1 for padding
        self.fitted = True

    def transform(self, text):
        if not self.fitted:
            raise ValueError("CharEncoder not fitted.")
        # unknown characters mapped to 0
        return np.array([self.char2id.get(c, 0) for c in text], dtype=np.int64)

    def fit_transform(self, texts):
        self.fit(texts)
        return [self.transform(t) for t in texts]


######## Define CNN Model #########


class DNASequenceCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_classes=2,
                 kernel_sizes=[3,5,7], num_filters=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])

        self.fc_entropy = nn.Linear(1, 16)
        self.fc = nn.Linear(len(kernel_sizes)*num_filters + 16, num_classes)

    def forward(self, x, entropy):
        """
        x: [batch, seq_len] integer-encoded text
        entropy: [batch, 1] global numeric feature
        """
        emb = self.embedding(x)           # [B, L, E]
        emb = emb.transpose(1,2)          # [B, E, L]

        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.adaptive_max_pool1d(c,1).squeeze(-1) for c in conv_outs]

        cnn_feat = torch.cat(pooled, dim=1)
        entropy_feat = F.relu(self.fc_entropy(entropy))
        feat = torch.cat([cnn_feat, entropy_feat], dim=1)

        return self.fc(feat)


######## Define DataLoader #########


class DNADataset(Dataset):
    def __init__(self, sequences, entropies, labels):
        self.sequences = sequences
        self.entropies = entropies
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.entropies[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


######## Training CNN Model #########

def train_CNN(X_train, y_train, save_weights=False):

            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Preprocessing
    texts_proc = [preprocess_text(t) for t in X_train]
    
    # Encode characters
    char_encoder = CharEncoder()
    sequences = char_encoder.fit_transform(texts_proc)
    
    # Pad sequences
    max_len = max(len(seq) for seq in sequences)
    sequences_pad = [np.pad(seq, (0, max_len - len(seq)), constant_values=0)
                     for seq in sequences]
    
    # Compute entropy features
    entropies_list = [[shannon_entropy(t)] for t in texts_proc]
    
    # Labels
    labels_list = y_train
    
    # generate dataloader
    dataset = DNADataset(sequences_pad, entropies_list, labels_list)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Define model, loss, optimiser
    model = DNASequenceCNN(vocab_size=char_encoder.vocab_size,
                           embed_dim=32,
                           num_classes=2).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop with Amp
    print("Start training...")
    for epoch in range(30):
        model.train()
        running_loss = 0.0
    
        for X_batch, entropy_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            entropy_batch = entropy_batch.to(device)
            y_batch = y_batch.to(device)
    
            optimizer.zero_grad()
    
            # Mixed precision context
            with torch.cuda.amp.autocast():
                logits = model(X_batch, entropy_batch)
                loss = criterion(logits, y_batch)
    
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            running_loss += loss.item() * X_batch.size(0)
    
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}")
    
    print("Training complete.")

    if(save_weights):
        torch.save(model.state_dict(), "models/CNN_model.pth")
        joblib.dump(char_encoder, "models/char_encoder.pkl")
        
    
