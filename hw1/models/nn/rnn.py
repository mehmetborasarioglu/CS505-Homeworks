import torch as pt
from collections.abc import Sequence, Mapping
from typing import Type, Tuple
import random
import os
import sys
from tqdm import tqdm

# Adjust the Python path so that project modules can be imported.
_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..", "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

# PYTHON PROJECT IMPORTS
from lm import LM, StateType
from vocab import Vocab, START_TOKEN, END_TOKEN

RNNType: Type = Type["RNN"]

class RNN(LM, pt.nn.Module):
    def __init__(self: RNNType,
                 data: Sequence[Sequence[str]],
                 hidden_size: int = 64,
                 num_epochs: int = 2,
                 saved_model_path: str = None) -> None:
        super().__init__()
        self.vocab = Vocab()
        
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)
        
        self.hidden_size = hidden_size
        self.rnn_cell = pt.nn.RNNCell(len(self.vocab), self.hidden_size)
        self.output_layer = pt.nn.Linear(self.hidden_size, len(self.vocab))
        
        if saved_model_path is None:
            o = pt.optim.Adam(self.parameters(), lr=1e-3)

            for epoch in range(num_epochs):

                # train the model
                random.shuffle(data) # Important

                train_chars = 0 # Total number of characters
                for line in tqdm(data, desc=f"epoch {epoch}"):
                    # Compute the negative log-likelihood of this line,
                    # which is the thing we want to minimize.
                    loss = 0.
                    q = self.start()
                    for c_in, c_out in zip([START_TOKEN] + line, line + [END_TOKEN]): # skip BOS
                        train_chars += 1
                        q, p = self.step(q, self.vocab.numberize(c_in))
                        loss -= p[self.vocab.numberize(c_out)]

                    # Compute gradient of loss with respect to parameters.
                    o.zero_grad()   # Reset gradients to zero
                    loss.backward() # Add in the gradient of loss

                    # Clip gradients (not needed here, but helpful for RNNs)
                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                    # Do one step of gradient descent.
                    o.step()
        else:
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

    def forward(self: RNNType, 
                w_idx: int, 
                h: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        x = pt.zeros(len(self.vocab))
        x[w_idx] = 1
        h = self.rnn_cell.forward(x, h)
        return h, pt.log_softmax(self.output_layer.forward(h),dim=0)

    def start(self: RNNType) -> StateType:
        return pt.zeros(self.hidden_size)  # shape will be (hidden_size,)


    def step(self: RNNType, 
             q: StateType, 
             w_idx: int) -> Tuple[StateType, Mapping[int, float]]:
        return self.forward(w_idx,q)
        


LSTMType: Type = Type["LSTM"]

class LSTM(LM, pt.nn.Module):
    def __init__(self: LSTMType,
                 data: Sequence[Sequence[str]],
                 hidden_size: int = 128,
                 num_epochs: int = 5,
                 saved_model_path: str = None) -> None:
        super().__init__()
        self.vocab = Vocab()
        
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)
        
        self.hidden_size = hidden_size
        self.lstm_cell = pt.nn.LSTMCell(len(self.vocab), self.hidden_size)
        self.output_layer = pt.nn.Linear(self.hidden_size, len(self.vocab))
        
        if saved_model_path is None:
            o = pt.optim.Adam(self.parameters(), lr=1e-3)

            for epoch in range(num_epochs):

                # train the model
                random.shuffle(data) # Important

                train_chars = 0 # Total number of characters
                for line in tqdm(data, desc=f"epoch {epoch}"):
                    # Compute the negative log-likelihood of this line,
                    # which is the thing we want to minimize.
                    loss = 0.
                    q, c = self.start()
                    for c_in, c_out in zip([START_TOKEN] + line, line + [END_TOKEN]): # skip BOS
                        train_chars += 1
                        (q, c), p = self.step((q, c), self.vocab.numberize(c_in))
                        loss -= p[self.vocab.numberize(c_out)]

                    # Compute gradient of loss with respect to parameters.
                    o.zero_grad()   # Reset gradients to zero
                    loss.backward() # Add in the gradient of loss

                    # Clip gradients (not needed here, but helpful for LSTMs)
                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                    # Do one step of gradient descent.
                    o.step()
        else:
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

    def forward(self: LSTMType, 
                w_idx: int, 
                h: pt.Tensor, 
                c: pt.Tensor) -> Tuple[Tuple[pt.Tensor, pt.Tensor], pt.Tensor]:
        x = pt.zeros(len(self.vocab))
        x[w_idx] = 1
        h, c = self.lstm_cell.forward(x, (h, c))
        return (h, c), pt.log_softmax(self.output_layer.forward(h), dim=0)

    def start(self: LSTMType) -> StateType:
        return pt.zeros(self.hidden_size), pt.zeros(self.hidden_size)  # shape will be (hidden_size,)

    def step(self: LSTMType, 
             q: StateType, 
             w_idx: int) -> Tuple[StateType, Mapping[int, float]]:
        h, c = q
        return self.forward(w_idx, h, c)