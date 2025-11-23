import os
from typing import Union, Tuple, Optional, Set
from pypdf import PdfReader
from docx import Document
import pandas as pd
import numpy as np
import string
import random
import torch

from preprocessing import *
from inference import *
from simple_check import *

import argparse
import os
import sys


# Add project root to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)
MODELS_DIR = os.path.join(ROOT, "models")



def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Process an input file.")
    
    parser.add_argument("file_path", type=str,
                        help="Absolute path to input file")
    parser.add_argument("--model", type=str, default='nlp',
                        help="Optional - model to be used ('rule-based','nlp','cnn')")

    args = parser.parse_args()

    # Validate that the file path is absolute
    if not os.path.isabs(args.file_path):
        print("ERROR: file_path must be an absolute path.")
        sys.exit(1)
    if not os.path.exists(args.file_path):
        print(f"ERROR: File does not exist: {args.file_path}")
        sys.exit(1)

    # Process file
    print(f"[INFO] Processing file: {args.file_path}")

    # Analyse file extension - flag True if genomic file type.
    flag_initial, file_ext, content = analyze_file(args.file_path)
    if (flag_initial):
        print(f"FILE CONTAINS GENOMIC DATA - Sensitive file extension detected ({file_ext}).")
        return True

    # Extract text and predict flag
    text = preprocess_text(content)

    if (args.model):
        model_type = args.model
    else:
        model_type = 'nlp'

    if (model_type=='nlp'):
        model_path = "../../models/NLP_model.pkl"
        # print("Model path -> ",model_path)
    elif (model_type=='cnn'):
        model_path = "../../models/CNN_model.pth"
        #model_path = os.path.join(MODELS_DIR, "CNN_model.pth")
    else:
        model_path = None
    
    pred = generate_predictions([text], model_type, model_path)[0]

    #print(pred)

    if(pred==1):
        print(f"FILE CONTAINS GENOMIC DATA - Sensitive content detected.")
        return True
    else:
        print(f"FILE DOES NOT CONTAIN GENOMIC DATA.")
        return True


if __name__ == "__main__":
    main()
 