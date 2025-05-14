
# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Type, Tuple
import argparse as ap
import numpy as np
import os
import sys
import torch as pt


# make sure the directory that contains this file is in sys.path
_cd_: str = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
from data.charloader import load_chars_from_file
from models.nn.rnn import RNN, LSTM
from vocab import START_TOKEN, END_TOKEN

def train_rnn() -> RNN:
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/large")

    saved_model_path = None
    if os.path.exists("./rnn.model"):
        saved_model_path = "./rnn.model"

    return RNN(train_data, saved_model_path=saved_model_path)

def dev_rnn(m: RNN) -> Tuple[int, int]:
    dev_data: Sequence[Sequence[str]] = load_chars_from_file("./data/dev")

    num_correct = 0
    total = 0
    for dev_line in dev_data:
        q = m.start()  # Initialize the hidden state

        for c_input, c_actual in zip([START_TOKEN] + dev_line, # read in string w/ <BOS> prepended
                                     dev_line + [END_TOKEN]):# check against string incl. <EOS>
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(p.argmax())

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total


def train_lstm() -> LSTM:
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/large")

    saved_model_path = None
    if os.path.exists("./lstm.model"):
        saved_model_path = "./lstm.model"

    return LSTM(train_data, saved_model_path=saved_model_path)

def dev_lstm(m: LSTM) -> Tuple[int, int]:
    data: Sequence[Sequence[str]] = load_chars_from_file("./data/dev")

    num_correct = 0
    total = 0
    for line in data:
        q, c = m.start()  # Initialize the hidden and cell state
        for c_input, c_actual in zip([START_TOKEN] + line, line + [END_TOKEN]):
            (q, c), p = m.step((q, c), m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(p.argmax().item())

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total

def test_lstm(m: LSTM) -> Tuple[int, int]:
    data: Sequence[Sequence[str]] = load_chars_from_file("./data/test")

    num_correct = 0
    total = 0
    for line in data:
        q, c = m.start()  # Initialize the hidden and cell state
        for c_input, c_actual in zip([START_TOKEN] + line, line + [END_TOKEN]):
            (q, c), p = m.step((q, c), m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(p.argmax().item())

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total


def main() -> None:
    m: LSTM = train_rnn()
    pt.save(m.state_dict(), "./rnn.model")

    num_correct, total = dev_rnn(m)

    print(num_correct / total)

if __name__ == "__main__":
    main()