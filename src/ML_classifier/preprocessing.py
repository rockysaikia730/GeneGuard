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
from simple_check import *


#################################### Identify file type ####################################

# A set of common bioinformatics/genomic file extensions.
GENOMIC_EXTENSIONS = [
    '.fastq', '.fq', '.fasta', '.fa', '.fna', '.gb', '.gff', '.gff3', '.gtf',
    '.sam', '.bam', '.cram', '.vcf', '.bcf', '.wig', '.bed', '.bigwig', '.tbi',
    '.tabix', '.h5', '.hdf5', # HDF5 often used for single-cell data (e.g., Anndata)
    ]

# A set of common document extensions for text extraction.
DOCUMENT_EXTENSIONS = ['.txt', '.pdf', '.doc', '.docx']

def analyze_file(file_path):
    """
    Analyzes a file path to determine if it is genomic data or a document,
    and returns the classification or the document content.

    Args:
        file_path: The full path to the file.

    Returns:
        A tuple (is_genomic_extension, file_extension, content_or_none_or_emptymessage).
    """
    if not os.path.exists(file_path):
        return (False, None, None)

    # Check for empty file before reading
    if os.path.getsize(file_path) == 0:
        return (False, None, "EMPTY FILE")

    # Get the file name and extension
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    # 1. Check for Genomic Data
    if ext in GENOMIC_EXTENSIONS:
        return (True, ext, None)

    # 2. Check for Document Data and Extract Content
    elif ext in DOCUMENT_EXTENSIONS:
        if ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if not content.strip():
                    return (False, ext, "Empty or non-text content")
                return (False, ext, content.strip())
            except Exception as e:
                return (False, ext, f"Could not read text file: {e}")

        elif ext == '.pdf':
            reader = PdfReader(file_path)
            content = "".join([page.extract_text() for page in reader.pages])
            if not content.strip():
                    return (False, ext, "Empty or non-text content")
            return (False, ext, content)

        elif ext in ['.doc', '.docx']:
            document = Document(file_path)
            content = "\n".join([paragraph.text for paragraph in document.paragraphs])
            if not content.strip():
                    return (False, ext, "Empty or non-text content")
            return (False, ext, content)

    # 3. Handle Unknown/Other Files
    return (False, ext, None)



#################################### Text preprocessing for test data ####################################


def unicode_dna_preprocessor(text: str) -> str:
    """
    Normalizes text by stripping invisible characters and transliterating 
    Unicode homoglyphs to ASCII to ensure clean genomic data.
    """
    # Standard Normalize (Fixes wide text 'Ａ' -> 'A')
    text = unicodedata.normalize('NFKC', text)
    
    # Remove Invisible Characters (Category Cf = Format, Cc = Control)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ["Cf", "Cc"])
    
    # Maps unicode to nearest ASCII
    text = unidecode(text)
    text = text.upper()

    return text


def preprocess_text(text):
    """
    Very minimal preprocessing:
    - Lowercase
    - Remove whitespace newlines
    We DO NOT filter characters, because DNA may be obfuscated.
    - Unicode detection processing
    """
    text = text.lower().replace("\n", " ").replace("\t", " ")
    text = unicode_dna_preprocessor(text)
    return text


# Entropy calculation
def shannon_entropy(text):
    """
    Compute Shannon entropy over all characters.
    """
    if len(text) == 0:
        return 0.0

    from math import log2
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1

    entropy = 0
    for c in freq:
        p = freq[c] / len(text)
        entropy -= p * log2(p)

    return entropy