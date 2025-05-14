from collections.abc import Sequence, Mapping
from typing import Type, Tuple
import collections
import numpy as np
import os
import sys

_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..", "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

# PYTHON PROJECT IMPORTS
from lm import LM, StateType
from vocab import Vocab, START_TOKEN, END_TOKEN

NgramType: Type = Type["Ngram"]

class Ngram(LM):
    def __init__(self, n: int, data: Sequence[Sequence[str]]) -> None:
        """
        Trains an n-gram language model on the given training 'data'.

        Arguments:
          n: The order of the n-gram model (e.g., 2=bigram, 3=trigram, etc.)
          data: A sequence of sentences (each sentence is a list of tokens).
        """
        self.n = n
        self.vocab = Vocab()
        # Store n-gram counts: ngram_counts[context][next_token]
        self.ngram_counts = collections.defaultdict(lambda: collections.defaultdict(int))

        # Absolute discount parameter; you can experiment with other values.
        self.discount = 0.5

        # Build the lower-order model recursively or use Unigram if n=2
        if n > 2:
            self.lower_ngram = Ngram(n - 1, data)
        else:
            self.lower_ngram = Unigram(data)
        
        for line in data:
            padded_line = [START_TOKEN] * (n - 1) + list(line) + [END_TOKEN]
            for token in padded_line:
                self.vocab.add(token)
            
            numberized_line = [self.vocab.numberize(t) for t in padded_line]
            for i in range(len(numberized_line) - (n - 1)):
                context = tuple(numberized_line[i : i + n - 1])   # (n-1)-tuple
                next_token = numberized_line[i + n - 1]
                self.ngram_counts[context][next_token] += 1

    
        self.log_probs = {}
        vocab_size = len(self.vocab)

        for context, next_tokens in self.ngram_counts.items():
            total_count = sum(next_tokens.values())
            distinct_next = len(next_tokens)  # n_1^+(u): number of distinct tokens after 'context'

            # lambda(u) = (d * n_1^+(u)) / sum_b c(b|u)
            if total_count > 0:
                lambda_u = (self.discount * distinct_next) / total_count
            else:
                lambda_u = 0.0

            # Retrieve the lower-order distribution or fallback to uniform.
            if self.n > 2:
                lower_context = tuple(context[1:])
                lower_log_probs = self.lower_ngram.log_probs.get(lower_context, None)
                if lower_log_probs is None:
                    # Fallback to uniform distribution over all vocab
                    lower_probs = np.full(vocab_size, 1.0 / vocab_size, dtype=float)
                    # Make sure START_TOKEN has zero probability
                    start_idx = self.vocab.numberize(START_TOKEN)
                    lower_probs[start_idx] = 0.0
                    # Renormalize if needed
                    prob_sum = lower_probs.sum()
                    if prob_sum > 0:
                        lower_probs /= prob_sum
                else:
                    lower_probs = np.exp(lower_log_probs)
            else:
                # If n=2, the lower model is Unigram
                lower_probs = np.exp(self.lower_ngram.logprob)

            # Now construct this context's distribution
            log_probs_array = np.empty(vocab_size, dtype=float)
            for token_id in range(vocab_size):
                c = next_tokens.get(token_id, 0)  # c(a|u)

                # Discounted higher-order part = max(c-d, 0) / total_count
                discounted_count = max(c - self.discount, 0)
                higher_order_prob = (discounted_count / total_count) if total_count > 0 else 0.0

                # Backoff part = lambda_u * P(a | lower_context)
                backoff_prob = lambda_u * lower_probs[token_id]

                p = higher_order_prob + backoff_prob
                log_probs_array[token_id] = np.log(p) if p > 0 else -np.inf

            # Important fix: Force START_TOKEN to -∞ for all contexts
            # so that we never predict <s> again after the beginning.
            start_idx = self.vocab.numberize(START_TOKEN)
            log_probs_array[start_idx] = -np.inf

            self.log_probs[context] = log_probs_array

    def start(self) -> np.ndarray:
        """
        Returns the model's initial state of length (n-1),
        which is all <s> tokens.
        """
        return np.array(
            [self.vocab.numberize(START_TOKEN) for _ in range(self.n - 1)],
            dtype=int
        )

    def step(self, q: StateType, w_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the old state (q) and a newly observed token (w_idx),
        return the new state and the distribution over the next token.
        """
        # Shift the context by 1 and append w_idx
        new_state = np.concatenate((q[1:], [w_idx]))
        context = tuple(new_state.tolist())
        vocab_size = len(self.vocab)

        # If context is known, use precomputed distribution
        if context in self.log_probs:
            logprob = self.log_probs[context]
        else:
            # Otherwise back off to lower-order model or uniform fallback
            if self.n > 2:
                lower_context = tuple(context[1:])
                lower_log_probs = self.lower_ngram.log_probs.get(lower_context, None)
                if lower_log_probs is None:
                    # Fallback to uniform distribution
                    # Exclude <s> (START_TOKEN) from having probability
                    logprob = np.full(vocab_size, -np.inf, dtype=float)
                    uniform_logprob = np.log(1.0 / (vocab_size - 1))

                    for token_id in range(vocab_size):
                        if token_id != self.vocab.numberize(START_TOKEN):
                            logprob[token_id] = uniform_logprob
                else:
                    logprob = lower_log_probs
            else:
                # For n=2, back off to the Unigram distribution
                logprob = self.lower_ngram.logprob

        return new_state, logprob



UnigramType: Type = Type["Unigram"]

class Unigram(LM):
    """A unigram language model with additive (Laplace) smoothing."""

    def __init__(self: UnigramType, data: Sequence[Sequence[str]]) -> None:
        self.vocab: Vocab = Vocab()
        count: collections.Counter = collections.Counter()
        total: int = 0
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)
                w_id = self.vocab.numberize(w)
                count[w_id] += 1
                total += 1

        alpha = 1.0  # smoothing parameter
        vocab_size = len(self.vocab)
        smoothed_total = total + alpha * (vocab_size-1)

        self.logprob = np.empty(vocab_size, dtype=float)
        for w_idx in range(vocab_size):
            if self.vocab.numberize(START_TOKEN) != w_idx:
                self.logprob[w_idx] = np.log((count[w_idx] + alpha) / smoothed_total)
            else:
                self.logprob[w_idx] = -np.inf
        print(self.logprob[self.vocab.numberize(START_TOKEN)])


    def start(self: UnigramType) -> StateType:
        return None

    def step(self: UnigramType, q: StateType, w_idx: int) -> Tuple[StateType, Mapping[str, float]]:
        return (None, self.logprob)