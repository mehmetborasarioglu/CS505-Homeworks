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


def main():
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")

    train_source_path = os.path.join(data_dir, "train.old")
    train_target_path = os.path.join(data_dir, "train.new")
    test_source_path = os.path.join(data_dir, "test.old")
    test_target_path = os.path.join(data_dir, "test.new")

    model = Modernizer().train(train_source_path, train_target_path,
                                          test_source_path=test_source_path,
                                          test_target_path=test_target_path,
                                          delta_coeff=10) #, limit=10)


if __name__ == "__main__":
    main()

