# Modernizer — Noisy‑Channel Spelling and Style Conversion
*CS‑505 / NLP — Homework 2*

## 1. Project Overview
This repository implements a complete noisy‑channel system for converting Early‑Modern English text to contemporary English. The implementation includes a finite‑state transducer (FST) for modeling orthographic variations, an n‑gram language model for scoring candidate outputs, and decoding routines based on the Viterbi algorithm with expectation‑maximization (EM) training variants.

## 2. Repository Structure
```
.
├── data/                 # Training and evaluation corpora (not included)
├── models/               # Package namespace for core modules
│   └── modernizer.py
├── fst.py                # Finite‑state transducer implementation
├── lm.py                 # n‑gram language model (uniform and Kneser–Ney)
├── vocab.py              # Vocabulary and token utilities
├── hard_em.py            # Hard‑EM training script
├── plot_hard_em.py       # Convergence plot for Hard‑EM (brittle variant)
├── soft_plot.py          # Soft‑EM training and plotting
├── part1.py              # Homework Part 1 experiments
├── part2.py              # Homework Part 2 experiments
├── part3.py              # Homework Part 3 experiments
├── init.py               # Minimal demonstration script
├── hw2_report_better.pdf # Technical report
└── README.md
```

## 3. Environment Setup
### 3.1 Prerequisites
* Python ≥ 3.9

Create a virtual environment and install required packages:
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
matplotlib
```

### 3.2 Dataset Layout
Place the corpus files in `./data/` with the following names:
```
data/
├── train.old   # Source sentences (Early‑Modern English)
├── train.new   # Target sentences (Contemporary English)
├── test.old
└── test.new
```

## 4. Training and Evaluation
| Variant          | Command                    | Output                                   |
|------------------|----------------------------|------------------------------------------|
| Hard EM          | `python hard_em.py`        | `saved_hard_em.pkl` and evaluation metrics |
| Soft EM          | `python soft_plot.py`      | `log_likelihood_soft.png`                |
| Brittled Hard EM | `python plot_hard_em.py`   | `log_likelihood_hard.png`                |

The trained model files are excluded from version control. Run the corresponding script to regenerate them.

## 5. Demonstration
```bash
python init.py
```
The script decodes the first ten lines of `test.old` and prints their modernized versions with associated log‑probabilities.

## 6. Reproducing Plots
To recreate learning‑curve figures:
```bash
python plot_hard_em.py   # Brittled Hard‑EM
python soft_plot.py      # Soft‑EM
```
Each script stores a PNG in the project root directory.

## 7. Results
| Training Regime | Character Error Rate (↓) | Iterations |
|-----------------|--------------------------|------------|
| Hard EM         | —                        | —          |
| Soft EM         | —                        | —          |
| Brittled Hard EM| —                        | —          |

Replace placeholder values after running the training scripts.

## 8. License
This repository is distributed under the MIT License.

## 9. Acknowledgments
Starter code provided by *CS‑505 Introduction to Natural Language Processing, Spring 2025.* All additional implementation and experimentation by Mehmet Bora Sarıoğlu.
