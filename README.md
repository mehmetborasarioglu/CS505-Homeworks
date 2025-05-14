# CS‑505 — Introduction to NLP  
Homework Solutions (Spring 2025)

This repository aggregates my solutions for all four homework assignments in CS‑505 / CAS CS595 at Boston University.  
Each homework is self‑contained inside its own directory with an independent README that documents implementation and usage details.

| Directory | Assignment Topic | Core Techniques |
|-----------|------------------|-----------------|
| **hw1** | *Statistical Language Modelling* ★ | n‑gram language models, perplexity evaluation, Laplace and Kneser–Ney smoothing |
| **hw2** | *Noisy‑Channel Spelling & Style Conversion* | Finite‑State Transducer (FST), n‑gram LM, Viterbi decoding, Hard/Soft EM |
| **hw3** | *Probabilistic Parsing with PCFGs* | Penn Treebank preprocessing, CNF transformation, CKY‑Viterbi, grammar induction |
| **hw4** | *Named Entity Recognition* | Hidden Markov Model, Structured Perceptron, BIO tagging, Viterbi decoding |

★ Homework 1 code and report are intentionally excluded from this public version due to data‑sharing restrictions mandated by the course.

---

## Repository Layout
```
.
├── hw1/                  # Statistical language modelling (see note above)
├── hw2/                  # Noisy‑channel modernizer
├── hw3/                  # PCFG CKY parser
├── hw4/                  # NER with HMM & perceptron
├── LICENSE
└── README.md             # ← current file
```

---

## Global Development Environment
Although each assignment can be executed in isolation, the following shared environment satisfies all dependencies:

### Minimum Requirements
* Python ≥ 3.9

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`
```
numpy
scipy          # HW3
matplotlib     # HW2, HW3 (plots)
tqdm
dill           # serializer used in HW2–HW4
```

> Some homeworks may introduce additional lightweight packages (e.g., `networkx` for optional extensions). Refer to the per‑homework README for task‑specific extras.

---

## How to Reproduce Results
1. `cd` into the homework directory of interest.  
2. Follow the steps outlined in that directory’s README (data download, preprocessing, training, evaluation).  
3. The scripts will generate artefacts under a `generated/` sub‑folder and print the required metrics specified by the assignment rubrics.

---

## License
The code is released under the MIT License (see `LICENSE`).  
Dataset files and starter skeletons remain under the original copyright of 
**CS‑505 Introduction to Natural Language Processing (Boston University)** and are **not** redistributed here.

---

