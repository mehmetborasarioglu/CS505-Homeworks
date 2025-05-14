# SYSTEM IMPORTS
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
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
from from_file import load_annotated_data
from layered_graph import LayeredGraph
from tables import BigramTable, EmissionTable, START_TOKEN, END_TOKEN, UNK_TOKEN


BaseType = Type["Base"]


class Base(object):
    def __init__(self: BaseType):
        self.lm: BigramTable = None
        self.tm: EmissionTable = None
        self.tag_vocab: Set[str] = set()
        self.word_vocab: Set[str] = set()

    def _init_lm(self: BaseType,
                 tag_corpus: Sequence[Sequence[str]],
                 init_val: float = 0.0
                 ) -> None:
        # two main differences for the alphabets of the tag model:
        #     1) The input alphabet contains the START symbol
        #     2) The output alphabet contains the STOP symbol
        tag_vocab = set()
        for tag_seq in tag_corpus:
            tag_vocab.update(tag_seq)

        self.tag_vocab = tag_vocab
        sorted_tags = sorted(tag_vocab)

        # personally I like to keep the alphabet sorted b/c it helps me read the table if I have to
        self.lm = BigramTable(sorted_tags, sorted_tags, init_val=init_val)

    def _init_tm(self: BaseType,
                 word_corpus: Sequence[Sequence[str]],
                 tag_corpus: Sequence[Sequence[str]],
                 init_val: float = 0.0
                 ) -> None:
        word_vocab = set()
        for word_seq, tag_seq in zip(word_corpus, tag_corpus):
            word_vocab.update(word_seq)

        # add UNK to the words
        self.word_vocab = word_vocab | set([UNK_TOKEN])

        # again I prefer to keep things sorted in case I need to try and reproduce something from lecture
        # when I'm debugging
        sorted_words = sorted(self.word_vocab)
        sorted_tags = sorted(self.tag_vocab)
        self.tm = EmissionTable(sorted_tags, sorted_words, init_val=init_val)

    def lm_count_bigram(self: BaseType,
                        tag_corpus: Sequence[Sequence[str]]
                        ) -> None:
        # TODO: complete me!
        #   iterate through each sequence of the corpus and increment the corresponding bigram entries
        for tag in tag_corpus:
            if tag: 
                self.lm.increment_value(START_TOKEN,tag[0])
                for i in range(1,len(tag)):
                    self.lm.increment_value(tag[i-1],tag[i])
                self.lm.increment_value(tag[len(tag)-1],END_TOKEN)
        #   don't forget to increment <EOS> after each sequence!

    def _train(self: BaseType,
               word_corpus: Sequence[Sequence[str]],
               tag_corpus: Sequence[Sequence[str]],
               init_val: float = 0.0
               ) -> BaseType:
        self._init_lm(tag_corpus, init_val=init_val)
        self._init_tm(word_corpus, tag_corpus, init_val=init_val)
        return self

    def train_from_raw(self: BaseType,
                       word_corpus: Sequence[Sequence[str]],
                       tag_corpus: Sequence[Sequence[str]],
                       limit: int = -1
                       ) -> BaseType:
        if limit > -1:
            word_corpus = word_corpus[:limit]
            tag_corpus = tag_corpus[:limit]
        return self._train(word_corpus, tag_corpus)

    def train_from_file(self: BaseType,
                        file_path: str,
                        limit: int = -1
                        ) -> BaseType:
        word_corpus, tag_corpus = load_annotated_data(file_path, limit=limit)
        return self._train_from_raw(word_corpus, tag_corpus, limit=limit)

    def parse_word_list(self: BaseType,
                        word_list: Sequence[str]
                        ) -> Sequence[str]:
        parsed_list: Sequence[str] = list(word_list)
        for i, w in enumerate(parsed_list):
            if w not in self.word_vocab:
                parsed_list[i] = UNK_TOKEN
        return parsed_list

    def viterbi_traverse(self,
                     word_list: Sequence[str],
                     init_func_ptr: Callable[[], LayeredGraph],
                     update_func_ptr: Callable[[str, str, str, float, LayeredGraph], None],
                     ) -> None:
        """
        Build the trellis layer by layer.  At each step we:
        - add a new layer
        - for each node in the old layer, pull its best path cost
        - call update_func_ptr with that cost
        """

        graph = init_func_ptr()
        graph.add_layer()
        graph.add_node(START_TOKEN, 0.0, None)

        for i, word in enumerate(word_list):
            graph.add_layer()
            for prev_tag in graph.node_layers[i]:
                prev_score, _ = graph.get_node_in_layer(i, prev_tag)
                for curr_tag in self.tag_vocab:
                    update_func_ptr(curr_tag, word, prev_tag, prev_score, graph)

        graph.add_layer()
        final_t = len(word_list)
        for prev_tag in graph.node_layers[-2]:
            prev_score, _ = graph.get_node_in_layer(final_t, prev_tag)
            update_func_ptr(END_TOKEN, END_TOKEN, prev_tag, prev_score, graph)





    def predict_sentence(self: BaseType,
                         word_list: Sequence[str]
                         ) -> Sequence[str]:
        word_list = self.parse_word_list(word_list)
        path, log_prob = self.viterbi(word_list)
        return path

    def predict(self: BaseType,
                word_corpus: Sequence[Sequence[str]]
                ) -> Iterable[Sequence[str]]:
        for word_list in word_corpus:
            yield self.predict_sentence(word_list)

