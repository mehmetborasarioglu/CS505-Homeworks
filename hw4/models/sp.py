# SYSTEM IMPORTS
from collections.abc import Callable, Sequence
from typing import Tuple, Type
from tqdm import tqdm
import numpy as np
import os
import sys
from random import shuffle
from collections import defaultdict 



_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from base import Base
from from_file import load_annotated_data
from layered_graph import LayeredGraph
from tables import START_TOKEN, END_TOKEN, UNK_TOKEN


class SP(Base):
    def __init__(self: Type["SP"]) -> None:
        super().__init__()

    def _init_tm(self: Type["SP"],
                 word_corpus: Sequence[Sequence[str]],
                 tag_corpus: Sequence[Sequence[str]],
                 init_val: float = 0.0
                 ) -> None:
        super()._init_tm(word_corpus, tag_corpus, init_val=init_val)

    def sp_training_algorithm(self: Type["SP"],
                              word_corpus: Sequence[Sequence[str]],
                              tag_corpus: Sequence[Sequence[str]],
                              ) -> Tuple[float, float]:
        
        
        idxs = list(range(len(word_corpus)))
        shuffle(idxs)

        num_correct = 0
        num_total   = 0
        batch_size  = 1

        for start in range(0, len(idxs), batch_size):
            batch = idxs[start:start + batch_size]


            delta_lm = defaultdict(float) 
            delta_tm = defaultdict(float) 

            for k in batch:
                gold_words = word_corpus[k]
                gold_tags  = tag_corpus[k]
                pred_tags  = self.predict_sentence(gold_words)
                
                num_total   += len(gold_tags)
                num_correct += sum(g == p for g,p in zip(gold_tags, pred_tags))

                prev_g, prev_p = START_TOKEN, START_TOKEN
                for g_tag, p_tag in zip(gold_tags, pred_tags):
                    delta_lm[(prev_g, g_tag)] += 1
                    delta_lm[(prev_p, p_tag)] -= 1
                    prev_g, prev_p = g_tag, p_tag
                delta_lm[(prev_g, END_TOKEN)] += 1
                delta_lm[(prev_p, END_TOKEN)] -= 1

                for w, g_tag, p_tag in zip(gold_words, gold_tags, pred_tags):
                    delta_tm[(g_tag, w)] += 1
                    delta_tm[(p_tag, w)] -= 1

            for (prev_tag, curr_tag), diff in delta_lm.items():
                self.lm.increment_value(prev_tag, curr_tag, diff)
            for (tag, word), diff in delta_tm.items():
                self.tm.increment_value(tag, word, diff)

        return num_correct, num_total
    
    def _train(self: Type["SP"],
               train_word_corpus: Sequence[Sequence[str]],
               train_tag_corpus: Sequence[Sequence[str]],
               dev_word_corpus: Sequence[Sequence[str]] = None,
               dev_tag_corpus: Sequence[Sequence[str]] = None,
               max_epochs: int = 20,
               converge_error: float = 1e-4,
               log_function: Callable[[Type["SP"], int, Tuple[int, int], Tuple[int, int]], None] = None
               ) -> Type["SP"]:
        super()._train(train_word_corpus, train_tag_corpus)

        current_epoch: int = 0
        current_accuracy: float = 1.0
        prev_accuracy: float = 1.0
        percent_rel_error: float = 1.0

        while current_epoch < max_epochs and percent_rel_error > converge_error:

            train_correct, train_total = self.sp_training_algorithm(train_word_corpus, train_tag_corpus)
            dev_correct, dev_total = 0, 0

            if dev_word_corpus is not None and dev_tag_corpus is not None:

                for i, predicted_tags in enumerate(self.predict(dev_word_corpus)):
                    true_tags = dev_tag_corpus[i]
                    dev_total += len(true_tags)
                    dev_correct += np.sum(np.array(true_tags) == np.array(predicted_tags))

            if log_function is not None:
                log_function(self, current_epoch, (train_correct, train_total), (dev_correct, dev_total))

            epoch_correct = train_correct if dev_word_corpus is None or dev_tag_corpus is None else dev_correct
            epoch_total = train_total if dev_word_corpus is None or dev_tag_corpus is None else dev_total

            prev_accuracy = current_accuracy
            current_accuracy = float(epoch_correct) / float(epoch_total)
            percent_rel_error = abs(prev_accuracy - current_accuracy) / prev_accuracy

            current_epoch += 1

        return self

    def train_from_raw(self: Type["SP"],
                       train_word_corpus: Sequence[Sequence[str]],
                       train_tag_corpus: Sequence[Sequence[str]],
                       dev_word_corpus: Sequence[Sequence[str]] = None,
                       dev_tag_corpus: Sequence[Sequence[str]] = None,
                       max_epochs: int = 20,
                       converge_error: float = 1e-4,
                       log_function: Callable[[int, Tuple[int, int], Tuple[int, int]], None] = None
                       ) -> None:
        self._train(train_word_corpus, train_tag_corpus,
                    dev_word_corpus=dev_word_corpus, dev_tag_corpus=dev_tag_corpus,
                    max_epochs=max_epochs,
                    converge_error=converge_error,
                    log_function=log_function)

    def train_from_file(self: Type["SP"],
                        train_path: str,
                        dev_path: str = None,
                        max_epochs=20,
                        converge_error: float = 1e-4,
                        limit: int = -1
                        ) -> None:
        train_word_corpus, train_tag_corpus = load_annotated_data(train_path, limit=limit)
        dev_word_corpus, dev_tag_corpus = None, None
        if dev_path is not None:
            dev_word_corpus, dev_tag_corpus = load_annotated_data(dev_path, limit=limit)
        self._train(train_word_corpus, train_tag_corpus,
                    dev_word_corpus=dev_word_corpus, dev_tag_corpus=dev_tag_corpus,
                    max_epochs=max_epochs,
                    converge_error=converge_error,
                    log_function=log_function)

    def viterbi(self: Type["SP"],
                word_list: Sequence[str]
                ) -> Tuple[Sequence[str], float]:

        # TODO: complete me!
        # This method should look nearly identical to your HMM viterbi!
        word_list = self.parse_word_list(word_list)

        graph = LayeredGraph(init_val=float("-inf"))

        def update(curr_tag: str,
                word: str,
                prev_tag: str,
                path_cost: float,
                graph: LayeredGraph
                ) -> None:
            trans_w = self.lm.get_value(prev_tag, curr_tag)
            emis_w  = self.tm.get_value(curr_tag, word)
            new_score = path_cost + trans_w + emis_w

            old_score, _ = graph.get_node_in_last_layer(curr_tag)
            if new_score > old_score:
                graph.add_node(curr_tag, new_score, prev_tag)

        self.viterbi_traverse(word_list, lambda: graph, update)

        last_layer = graph.node_layers[-1]
        best_score, parent = last_layer[END_TOKEN]

        path = []
        curr_tag = parent
        for layer in reversed(graph.node_layers[:-1]):
            if curr_tag == START_TOKEN:
                break
            path.append(curr_tag)
            _, curr_tag = layer[curr_tag]
        path.reverse()

        return path, best_score

