# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.dirname(os.path.abspath(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from models.modernizer import Modernizer
from create_model_for_string import create_model_for_string


def parta(data_dir):
    return Modernizer().train_language_model(os.path.join(data_dir, "train.new"))


def partb(model, data_dir):
    return model.init_typo_model(os.path.join(data_dir, "train.old"),
                                 os.path.join(data_dir, "train.new"), limit=1)


def partc(s):
    return create_model_for_string(s)


def main():
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")

    m1 = parta(data_dir)
    m2 = partb(m1, data_dir)
    # m2.visualize_typo_model()
    m3 = partc("hello")
    m3.visualize()
    


if __name__ == "__main__":
    main()

