
from collections.abc import Sequence
from typing import Tuple
import numpy as np
import os
import sys

# Ensure the project modules can be imported
_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_

# PYTHON PROJECT IMPORTS
from data.charloader import load_chars_from_file
from models.ngram.ngram import Ngram
from vocab import START_TOKEN, END_TOKEN


def train_ngram() -> Ngram:
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/large")
    return Ngram(n=5, data=train_data)


def dev_ngram(m: Ngram) -> Tuple[int, int]:
    dev_data: Sequence[Sequence[str]] = load_chars_from_file("./data/dev")

    num_correct: int = 0
    total: int = 0
    for dev_line in dev_data:
        q = m.start()

        for c_input, c_actual in zip([START_TOKEN] + dev_line, dev_line + [END_TOKEN]):
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(np.argmax(p))

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total

def test_ngram(m: Ngram) -> Tuple[int, int]:
    test_data: Sequence[Sequence[str]] = load_chars_from_file("./data/test")

    num_correct: int = 0
    total: int = 0
    for test_line in test_data:
        q = m.start()

        for c_input, c_actual in zip([START_TOKEN] + test_line, test_line + [END_TOKEN]):
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(np.argmax(p))

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total



def main() -> None:
    m: Ngram = train_ngram()
    num_correct, total = dev_ngram(m)
    print(f"Development Set Accuracy: {num_correct / total:.4%}")


if __name__ == "__main__":
    main()