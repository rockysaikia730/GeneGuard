# 🧬 Gene-Guard: Real-Time Genomic Data Leak Prevention 

**Gene Guard** is a Python tool designed to parse and flag data files containing genomic data using machine learning (NLP, CNN) and rule-based algorithms.

## Installation

You can install the package directly via pip:

```bash
pip install gene-guard==0.1.0
````

-----

## Usage

### 1\. Python Library

You can import `gene-guard` into your Python scripts to perform classification programmatically.

```python
from gene_guard.parse_and_classify import parse_and_classify

# Define the path to your input file
filepath = "path/to/your/data.txt"

# Run the classifier
# Available models: "nlp", "cnn", "rule-base"
parse_and_classify(filepath, model="nlp")
```

### 2\. Command Line Interface (CLI)

If you have cloned the repository, you can run inference directly from the terminal using the parser script.

**Syntax:**

```bash
python ./src/ML_clasifier/run_parser.py <filepath> --model <model_type>
```

-----

## Available Models

| Model Name | Argument (`--model`) | Description |
| :--- | :--- | :--- |
| **NLP** | `nlp` | Uses Natural Language Processing techniques for classification. |
| **CNN** | `cnn` | Uses a Convolutional Neural Network architecture. |
| **Rule Based** | `rule-base` | Uses heuristic rules to parse and classify the file. |

-----

