# PCFG CKY Parser — ATIS Penn Treebank  
*CS‑505 / NLP — Homework 3*

## 1. Project Overview
This repository provides an end‑to‑end implementation of a probabilistic context‑free grammar (PCFG) parser trained on the ATIS subset of the Penn Treebank.  
The system includes:

| Module | Purpose |
|--------|---------|
| **preprocess.py** | Converts Penn Treebank trees to Chomsky‑normal form (CNF). |
| **postprocess.py** | Reverses CNF transformations, restoring human‑readable trees. |
| **unknown.py** | Replaces singleton terminal symbols with the special token `<unk>`. |
| **trees.py** | Lightweight tree data structure with binarisation and unit‑rule utilities. |
| **models/grammar.py** | PCFG representation and probability normalisation. |
| **models/parser.py** | CKY and CKY‑Viterbi parsing with back‑pointer reconstruction. |
| **part1.py** | Grammar extraction, rule counting, probability estimation (Homework Part 1). |
| **part2.py** | CKY‑Viterbi parsing, runtime analysis, evaluation against gold trees (Homework Part 2). |
| **part3.py** | Placeholder for parser improvements (Homework Part 3). |
| **eval/** | Evaluation utilities (EVALB‑style F1 computation). |
| **generated/** | Auto‑generated files (pre‑processed trees, parses, plots). |

The assignment specification is included in **`hw3.pdf`** for reference.

## 2. Repository Structure
```
.
├── data/                 # ATIS corpus (not provided)
│   ├── train.trees
│   ├── dev.strings
│   ├── dev.trees
│   ├── test.strings
│   └── test.trees
├── generated/            # Output directory (created by scripts)
├── eval/                 # Evaluation scripts (eval/evalb.py, ...)
├── models/
│   ├── grammar.py
│   └── parser.py
├── preprocess.py
├── postprocess.py
├── unknown.py
├── trees.py
├── part1.py
├── part2.py
├── part3.py              # Optional extensions
├── hw3.pdf
└── README.md
```

## 3. Environment Setup
### 3.1 Prerequisites
* Python ≥ 3.9

### 3.2 Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`
```
numpy
scipy
matplotlib
```

## 4. Data Preparation
1. **Create an output directory**  
   ```bash
   mkdir -p generated
   ```
2. **Binarise and remove unary rules**  
   ```bash
   python preprocess.py data/train.trees -o generated/train.trees.pre
   ```
3. **Verify reversibility**  
   ```bash
   python postprocess.py generated/train.trees.pre -o generated/train.trees.post
   diff data/train.trees generated/train.trees.post   # should be empty
   ```
4. **Replace singletons with `<unk>`**  
   ```bash
   python unknown.py generated/train.trees.pre -o generated/train.trees.pre.unk
   ```

## 5. Running the Assignment Scripts
### 5.1 Grammar Extraction (Part 1)
```bash
python part1.py
```
Outputs the number of unique rules, the top‑N frequent rules, conditional probabilities, and a saved grammar.

### 5.2 CKY‑Viterbi Parsing and Evaluation (Part 2)
```bash
python part2.py
```
* Generates `generated/dev.parses`, runs post‑processing, and reports F1 on the development set.  
* Measures CKY‑Viterbi runtime per sentence, writes a log–log runtime plot, and estimates the empirical exponent *k*.

### 5.3 Parser Enhancements (Part 3)
Implement improvements in `part3.py`; the script should print new dev‑set F1 scores and produce `generated/test.parses` for the held‑out test set.

## 6. Expected Results
Target scores required by the assignment:
| Metric | Threshold |
|--------|-----------|
| Dev‑set F1 (Part 2) | ≥ 88 % |
| Test‑set F1 (Part 3, after improvements) | ≥ 90 % |

## 7. License
Released for educational use under the MIT License.

## 8. Acknowledgments
Starter code supplied by **CS‑505 Introduction to Natural Language Processing, Spring 2025**. All additional implementation and experimentation by Mehmet Sarioglu.
