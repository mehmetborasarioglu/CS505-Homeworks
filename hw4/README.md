# BIO‑Tagging with HMM and Structured Perceptron  
*CS‑505 / NLP — Homework 4*

## 1. Project Overview
This repository contains an end‑to‑end implementation of two sequence‑labeling models for Named Entity Recognition (NER) on the noisy Twitter NER dataset provided with the assignment:

1. **Hidden Markov Model (HMM)** — Part 1  
   *Supervised, first‑order model trained by relative‑frequency estimation with add‑0.1 smoothing; decoded with Viterbi.*

2. **Structured Perceptron (SP)** — Part 2  
   *Discriminative model sharing the HMM topology; trained with the online perceptron algorithm and decoded with Viterbi.*

Part 3 is left for user‑defined extensions (additional features, averaged weights, etc.).

The assignment specification is available in `hw4.pdf`; the submitted report is `hw4_submission.pdf`.

---

## 2. Repository Structure
```
.
├── data/                 # Training / dev / test files (not included)
│   ├── train
│   ├── dev
│   └── test
├── eval/                 # Evaluation scripts (conlleval wrapper)
│   ├── conlleval.pl
│   └── do_eval.py
├── generated/            # Auto‑generated outputs (created by scripts)
├── models/
│   ├── base.py
│   ├── hmm.py
│   └── sp.py
├── from_file.py          # Data‑loading utilities
├── layered_graph.py      # Layered graph abstraction for dynamic‑programming traversals
├── tables.py             # Dense numpy tables for bigram and emission probabilities
├── part1.py              # HMM training, inspection, and dev‑set evaluation
├── part2.py              # Structured perceptron training and dev‑set evaluation
├── part3.py              # (Optional) model improvements
├── hw4.pdf               # Assignment brief
├── hw4_submission.pdf    # Report
└── README.md
```

---

## 3. Environment Setup
### 3.1 Prerequisites
* Python ≥ 3.9
* Perl (needed for `conlleval.pl` inside `eval/`)

### 3.2 Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`
```
numpy
tqdm
dill
```

---

## 4. Data Preparation
Download the `train`, `dev`, and `test` files supplied with the homework and place them under `./data/`.  
Each file contains one token per line with optional BIO tags; blank lines mark sentence boundaries.

---

## 5. Running the Assignment Scripts
| Task | Command | Output |
|------|---------|--------|
| **Summary statistics** | `python part1.py` (Part A) | Prints counts of tokens, vocabulary, and tag types. |
| **Train + evaluate HMM** | `python part1.py` (Parts B–C) | Saves `hmm.pkl`; writes `generated/dev.out`; prints dev F₁. |
| **Train + evaluate SP** | `python part2.py` | Saves `sp.pkl`; writes `generated/dev.out`; prints dev F₁. |
| **Extensions** | Implement in `part3.py` | Should load best model and output test predictions. |

Both `part1.py` and `part2.py` will reuse saved model pickles if present, avoiding retraining.

---

## 6. Expected Results
Minimum dev‑set performance required by the assignment:

| Model | Accuracy | F₁ (overall) |
|-------|----------|--------------|
| HMM   | ≥ 70 %   | ≥ 9 % |
| SP    | ≥ 90 %   | ≥ 15 % |

Replace these placeholders with actual numbers after running the scripts on your environment.

---

## 7. Checkpoint Files
The repository ignores large binary checkpoints (`hmm.pkl`, `sp.pkl`) via `.gitignore`.  
If these files are missing, re‑run the training commands above.

---

## 8. License
Released under the MIT License.

---

## 9. Acknowledgments
Starter code and dataset courtesy of **CS‑505 Introduction to Natural Language Processing, Spring 2025**.  
Implementation and experimentation by *Mehmet Bora Sarıoğlu*.
