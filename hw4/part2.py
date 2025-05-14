# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
import itertools
import os
import sys
from tqdm import tqdm
import pickle


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "models"), os.path.join(_cd_, "eval")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from from_file import load_annotated_data
from eval.do_eval import write_output_and_evaluate
from models.hmm import HMM
from models.sp import SP


def part_a(train_word_corpus: Sequence[Sequence[str]],
           train_tag_corpus: Sequence[Sequence[str]],
           generated_dir: str,
           dev_word_corpus: Sequence[Sequence[str]] = None,
           dev_tag_corpus: Sequence[Sequence[str]] = None
           ) -> SP:
    model = SP()
    dev_out_path = os.path.join(generated_dir, "dev.out")

    pickle_path = os.path.join("sp.pkl")

    # if we already trained and saved it, just load it
    if os.path.exists(pickle_path):
        print(f"Loading existing SP model from {pickle_path}")
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)
        return model

    def log_function(model: SP,
                     epoch_num: int,
                     train_results: Tuple[int, int],
                     dev_results: Tuple[int, int]
                     ) -> None:
        # measure eval predictions
        if dev_word_corpus is not None and dev_tag_corpus is not None:
            dev_predicted_corpus: Sequence[Sequence[str]] = list()
            for predicted_seq in model.predict(dev_word_corpus):
                dev_predicted_corpus.append(predicted_seq)

            perf_path = os.path.join(generated_dir, f"epoch-{epoch_num}.result")
            print(f"writing results to {perf_path}")
            write_output_and_evaluate(dev_out_path, dev_word_corpus, dev_predicted_corpus, dev_tag_corpus,
                                      outfile=perf_path)

    model.train_from_raw(train_word_corpus, train_tag_corpus,
                         dev_word_corpus=dev_word_corpus, dev_tag_corpus=dev_tag_corpus, 
                         converge_error=1e-7, max_epochs=300, log_function=log_function)
    
    with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)


    return model


def part_b(model: SP,
           dev_out_path: str,
           dev_word_corpus: Sequence[Sequence[str]],
           dev_tag_corpus: Sequence[Sequence[str]]
           ) -> None:
    # do predictions
    print_cap: int = 5
    dev_predicted_corpus = list()
    for i, predicted_seq in enumerate(model.predict(dev_word_corpus)):
        dev_predicted_corpus.append(predicted_seq)
        if i < print_cap:
            for w, p, a in zip(dev_word_corpus[i], dev_tag_corpus[i], predicted_seq):
                print(w, p, a)
            print()

    write_output_and_evaluate(dev_out_path, dev_word_corpus, dev_predicted_corpus, dev_tag_corpus)


def main():
    # data
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    train_data_path = os.path.join(data_dir, "train")
    dev_data_path = os.path.join(data_dir, "dev")

    # generated files
    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    dev_out_path = os.path.join(generated_dir, "dev.out")

    train_word_corpus, train_tag_corpus = load_annotated_data(train_data_path)
    dev_word_corpus, dev_tag_corpus = load_annotated_data(dev_data_path)

    # Load existing SP model if available, otherwise train and save
    
    model = part_a(train_word_corpus, train_tag_corpus, generated_dir,
                       dev_word_corpus=dev_word_corpus, dev_tag_corpus=dev_tag_corpus)
    # Evaluate on dev set
    part_b(model, dev_out_path, dev_word_corpus, dev_tag_corpus)


if __name__ == "__main__":
    main()
