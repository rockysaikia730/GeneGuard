# 🧬 Gene-Guard: Real-Time Genomic Data Leak Prevention 

**Gene Guard** is a Python tool designed to parse and flag data files containing genomic data using machine learning (NLP, CNN) and rule-based algorithms. Beyond simple detection, it is capable of identifying and extracting genetic sequences embedded within mixed or obfuscated text.

## Installation

You can install the package directly via pip:

```bash
pip install gene-guard==0.1.0
````
For full package details, see the [Gene-Guard PyPI page](https://pypi.org/project/gene-guard/).

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

If you have cloned the repository, you can run inference directly from the terminal using the parser script. The file path entered must be absolute, and optionally the model type may be entered ('nlp', 'cnn', or 'rule-base').

**Syntax:**

```bash
python ./src/ML_clasifier/run_parser.py <filepath> --model <model_type>
```

-----

## Available Models

| Model Name | Argument (`--model`) | Description |
| :--- | :--- | :--- |
| **NLP** | `nlp` | Uses Natural Language Processing techniques for classification. |
| **CNN** | `cnn` | Uses a 1-D Convolutional Neural Network architecture. |
| **Rule Based** | `rule-base` | Uses heuristic rules to parse and classify the file. |

-----

#### Risk Detection Demo

To check the risk level of a specific text sequence using a trained checkpoint, use the demo script:

```bash
python genomic_detector_demo.py --checkpoint ./outputs/checkpoints/after_hard --threshold <Threshold> --text <text>
```
Example
```bash
python genomic_detector_demo.py --checkpoint ../../outputs/checkpoints/after_hard --threshold 0.01 --text "Data stream contains: aTgC gAtC gAtC gAtC embedded in the output lo" 
```
## Dataset

For detailed information regarding the datasets used for training, testing, and validation of these models, please refer to **[DATASET.md](https://github.com/rockysaikia730/GeneGuard/blob/main/data/DATASET.md)**.

## Technical Report

For a comprehensive overview of the system architecture and methodology, please refer to the project report:
**[Gene Guard: Real-Time Genomic Data Leak Prevention](https://github.com/rockysaikia730/GeneGuard/blob/main/docs/Gene%20Guard_%20Real-Time%20Genomic%20Data%20Leak%20Prevention.pdf)**
