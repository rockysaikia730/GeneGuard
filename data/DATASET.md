## 1. Raw Data Acquisition

### Source
The raw genomic data was sourced from the **National Center for Biotechnology Information (NCBI)** database. We specifically started with bacteria within the *Escherichia* genus.

### Download Protocol
The data was retrieved using the `ncbi-genome-download` tool. The specific command executed for the acquisition was:

```bash
ncbi-genome-download bacteria --genera Escherichia --assembly-levels complete --formats fasta
```

## 2. Dataset Creation & Sampling Strategy

### Preprocessing
The aggregated raw data was processed into a single text file named `combined_bacteria_sequences.txt`.
* **Content:** This file contains the full gene sequences for the *Escherichia* genus.
* **Structure:** Sequences belonging to different species are strictly separated by newlines, ensuring each line represents a distinct genomic entry.

### Objective:

The primary objective of this dataset is to facilitate the testing of a flagging system against various leakage vectors. We address scenarios where sensitive genomic data is written into documents within a monitored system (DLP). These leaks may stem from:

* **Malicious Exfiltration:** Human actors or autonomous **AI agents** attempting to steal data by bypassing existing DLP controls as they can't track biological data.
* **Employee Negligence:** Accidental data exposure, such as employees copy-pasting sequences into notes or emails for debugging purposes without realizing the security violation.

We define "sensitive data" in this context as biologically significant DNA fragments, which typically fall within the range of **100 to 500 base pairs (bp)**.

### Sampling Methodology
To generate the dataset used for training/testing:

1.  **Sequence Selection:** A specific species (line) is randomly selected from `combined_bacteria_sequences.txt`.
2.  **Length Determination:** A target length is randomly determined between **100 and 500 bp**.
3.  **Extraction:** A random starting index is selected within the chosen sequence. A substring of the determined length is then extracted from that index.

## 3. Adversarial Dataset Curation & Methodology

To ensure the flagging system is robust against active exfiltration attempts, we moved beyond raw DNA sequences. We curated a synthetic dataset that simulates various "adversarial" techniques an attacker might use to hide genomic data within standard business documents.

### Class Definitions
* **Positive (Flag):** The text contains sensitive DNA sequences, either in raw form, embedded in English text, or obfuscated using specific evasion techniques.
* **Negative (Benign):** The text contains standard English, random ASCII noise, or "fake" DNA-like sequences that do not carry biological significance.

### Data Generation Strategies
We employed 11 distinct strategies to generate samples. These strategies mix biologically valid *Escherichia* sequences with semantically meaningful English text derived from LLMs or random noise.

| ID | Strategy Name | Description | 
| :--- | :--- | :--- | 
| **1** | **Semantic Embedding** | Contiguous DNA sequences inserted into meaningful English sentences. <br>*(Includes: Raw DNA, DNA+English, Pure English)* |
| **2** | **Random Text Embedding** | Contiguous DNA sequences inserted into random ASCII text strings. | 
| **3** | **Noise Injection** | DNA sequences interrupted by alternate random characters (e.g., `A#T%G@C`). | 
| **4** | **Run-Length Encoding (RLE)** | Compression evasion where repeats are numbered (e.g., `aaaagccctt` $\rightarrow$ `a4gc3t`). | 
| **5** | **Compressed & Replaced** | RLE compression applied, followed by mapping ACGT to random single ASCII characters. | 
| **6** | **Character Substitution** | Simple cipher where A, C, G, T are randomly mapped to other ASCII characters. | 
| **7** | **Fake DNA (Control)** | Randomly generated sequences using only {A, C, G, T} to teach the model to distinguish real genomic patterns from random noise. |
| **8** | **Multi-Character Mapping** | A single DNA base is mapped to a sequence of multiple ASCII characters. | 
| **9** | **Random Spacing/Character** | DNA sequences obfuscated by inserting random length whitespaces/character between bases (non-contiguous). | 
| **10** | **Fragmentation** | DNA sequences are broken into smaller chunks and dispersed throughout English text (non-contiguous). | 
| **11** | **Random Noise** | Purely randomly generated ASCII characters (High volume to balance the dataset). |

### Final Distribution & Split
To prevent class imbalance bias, the noise generation (Strategy 11) was dynamically calculated to ensure a perfect 50/50 split between positive and negative samples.

* **Total Samples:** 11,600
* **Positive (Flag):** 5,800 (50%)
* **Negative (Benign):** 5,800 (50%)

**Train/Test Split:**
The dataset uses a stratified split to ensure distributions remain consistent across sets.
* **Training Set:** 80%
* **Testing Set:** 20%

## 4. Out-of-Distribution (OOD) Test Set

To evaluate the model's generalization capabilities and ensure it has not simply memorized the specific patterns of the *Escherichia* genus, we curated a completely separate test set using a distinct biological source.

**Selection: *Clostridium botulinum***
We selected *Clostridium botulinum*, a bacterium phylogenetically distinct from *Escherichia*.
* **Reasoning:** As a potent toxin-producing organism, it represents a high-value target for data exfiltration. Using a different genus ensures the model can flag sensitive genomic sequences regardless of the specific organism.

**Acquisition Command**
```bash
datasets download genome taxon "Clostridium botulinum" --reference --include genome,gff3 --filename botulinum_genomes.zip
```
### 4.1 Adversarial Dataset Curation & Methodology

To ensure the flagging system is robust against active exfiltration attempts, we created new adversarial attacks. We curated a dataset,stored in **`final_test_adversarial.pkl`**, containing **10 specific adversarial attack vectors**. These strategies simulate techniques an attacker might use to bypass pattern-matching (Regex) or simple DLP filters.

### Adversarial Attack Strategies
The positive class was generated using the following distinct obfuscation and steganography techniques:

| Strategy Name | Description |
| :--- | :--- |
| **1. Reverse Complement** | **Biological Obfuscation:** The sequence is transformed into its reverse complement (e.g., $5'-ATCG-3' \rightarrow 3'-TAGC-5'$), a common biological format that simple string matching might miss. |
| **2. Zero-Width Injection** | **Signature Breaking:** Non-printing characters (e.g., `U+200B` Zero Width Space) are inserted randomly between DNA bases to break contiguous text signatures while remaining invisible to the human eye. |
| **3. Columnar Split** | **Layout Manipulation:** The DNA sequence is split vertically across multiple lines or columns (e.g., imitating a CSV or columnar log), preventing linear reading. |
| **4. Linguistic Camouflage** | **Steganography:** The DNA sequence is hidden within natural-sounding English text using two distinct sub-methods:<br>• **Acrostic (70%):** The first letter of each word corresponds to a DNA base.<br>• **Codon Words (30%):** Actual words that double as genetic codons (e.g., "The **CAT** sat...") are used to construct the sequence. |
| **5. Multicharacter Mapping** | **Cipher:** Each DNA base is mapped to a specific sequence of multiple random ASCII characters. |
| **6. Compressed Sequence** | **Encoding:** The sequence is compressed using Run-Length Encoding or similar schemes (e.g., `A4G3`) to alter the visual pattern. |
| **7. Log/JSON Format** | **Context Masking:** The sequence is embedded within the values of a structured JSON object or fake server log, mimicking harmless technical data. |
| **8. Alternate Random Char** | **Noise Injection:** Random ASCII characters are interleaved between valid DNA bases (e.g., `A#T%G`), breaking adjacency while retaining order. |
| **9. Base64 Encoding** | **Standard Encoding:** Base64 regroups the binary data of the DNA sequence by splitting every three 8-bit bytes into four 6-bit chunks, which are then mapped to a standard alphanumeric set. This transformation completely masks recognizable biological patterns (like turning ACG into QUNH), allowing the data to bypass standard regex detection filters. |
| **10. Unicode Homoglyphs** | **Visual Spoofing:** DNA characters (A, C, G, T) are replaced with look-alike characters from different alphabets (e.g., Cyrillic 'а' vs. Latin 'a'), which look identical to humans but have different byte values. |