# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.dirname(os.path.abspath(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from eval.cer import cer
from models.modernizer import Modernizer
from from_file import load_mono


def parta(data_dir, limit=None):
    train_new_path = os.path.join(data_dir, "train.new")
    train_old_path = os.path.join(data_dir, "train.old")
    return Modernizer().train_language_model(train_new_path).init_typo_model(train_old_path, train_new_path, limit=limit)


def partbcd(model, data_dir):
    test_old_path = os.path.join(data_dir, "test.old")
    # for line, log_prob in model.decode(test_old_path, limit=10):
    #     print("%s\t%s" % (line, log_prob))

def parte(model, data_dir):
    test_old_path = os.path.join(data_dir, "test.old")
    test_new_path = os.path.join(data_dir, "test.new")

    # def line_filter(line):
    #     return line.rstrip()
    # new_data = model._read_data_from_file(test_new_path, line_filter_func=line_filter)

    new_data = load_mono(test_new_path)
    decodings = model.decode(test_old_path)

    formatted_data = list()
    for i, (line, log_prob) in enumerate(decodings):
        if i < 10:
            print("%s\t%s" % (line, log_prob))
        formatted_data.append((new_data[i], line))
    print("len formatted data: %s" % len(formatted_data))
    error_rate = cer(formatted_data)
    print(error_rate)


def main():
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")

    print("creating and intializing model")
    model = parta(data_dir)
    # model._typo_model.visualize()
    # print("number of language model states: %s" % len(model._language_model.states))
    # print("number of typo model states: %s" % len(model._typo_model.states))

    # print("topological order for typo model: %s" % (hw2.sorters.fst_topological_sort(model._typo_model)))
    # print("topological order for 'hello' model: %s" %
    #     (hw2.sorters.fst_topological_sort(hw2.models.create_model_for_string("hello"))))

    print("testing on first 10 lines")
    partbcd(model, data_dir)

    # print("computing error rate for the whole file")
    parte(model, data_dir)


if __name__ == "__main__":
    main()

