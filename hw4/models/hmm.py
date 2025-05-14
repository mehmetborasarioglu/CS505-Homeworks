# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Type, Tuple
import numpy as np
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from base import Base
from layered_graph import LayeredGraph
from tables import START_TOKEN, END_TOKEN, UNK_TOKEN


class HMM(Base):
    def __init__(self: Type["HMM"]) -> None:
        super().__init__()

    def _train_lm(self: Type["HMM"],
                  tag_corpus: Sequence[Sequence[str]]
                  ) -> None:
        self.lm_count_bigram(tag_corpus)
        self.lm.normalize_cond()

    def _train_tm(self,
              word_corpus: Sequence[Sequence[str]],
              tag_corpus: Sequence[Sequence[str]]
              ) -> None:
        for word_seq, tag_seq in zip(word_corpus, tag_corpus):
            for w, t in zip(word_seq, tag_seq):
                self.tm.increment_value(t, w, 1)
            self.tm.increment_value(END_TOKEN, END_TOKEN, 1)

        self.tm.normalize_cond(add=0.1)


    def _train(self: Type["HMM"],
               word_corpus: Sequence[Sequence[str]],
               tag_corpus: Sequence[Sequence[str]]
               ) -> Type["HMM"]:
        super()._train(word_corpus, tag_corpus)
        self._train_lm(tag_corpus)
        self._train_tm(word_corpus, tag_corpus)
        return self

    def viterbi(self: Type["HMM"],
                word_list) -> Tuple[Sequence[str], float]:
        # TODO: complete me!
        word_list = self.parse_word_list(word_list)
        graph = LayeredGraph(init_val=float("-inf"))

        def update(curr_tag: str,
           word: str,
           prev_tag: str,
           path_cost: float,
           graph: LayeredGraph
          ) -> None:
            trans_prob = self.lm.get_value(prev_tag, curr_tag)
            emis_prob  = self.tm.get_value(curr_tag, word)

            if trans_prob == 0 or emis_prob == 0:
                return

            new_score = path_cost + np.log(trans_prob) + np.log(emis_prob)

            curr_layer = graph.node_layers[-1]
            old_score, _ = curr_layer[curr_tag]
            if new_score > old_score:
                graph.add_node(curr_tag, new_score, prev_tag)


        self.viterbi_traverse(word_list, lambda: graph, update)

        last_layer = graph.node_layers[-1]
        best_final_tag = max(last_layer.items(), key=lambda x: x[1][0])[0]
        best_score, parent = last_layer[best_final_tag]  

        path = []
        curr_tag = parent  

    
        for layer in reversed(graph.node_layers[:-1]):  
            if curr_tag == "<BOS>":
                break
            path.append(curr_tag)
            _, parent = layer[curr_tag]
            curr_tag = parent

        path.reverse()
        return path, best_score


        # this method should return the most probable path along with the logprob of the most probable path

