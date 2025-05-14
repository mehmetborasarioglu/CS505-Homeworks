import os
import sys
from from_file import load_mono
from models.modernizer import Modernizer  # or wherever your Modernizer is defined
import numpy as np

def main():
    this_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(this_dir, "data")

    old_path = os.path.join(data_dir, "train.old")
    new_path = os.path.join(data_dir, "train.new")

    s_data = load_mono(old_path)  # Each line is a string, e.g., "thou art"
    w_data = load_mono(new_path)  # Each line is a string, e.g., "you are"

    mod = Modernizer()

    mod.init_typo_model(s_data, w_data)

    mod.visualize_tm()

    print("Typo model has", len(mod.tm.states), "states.")

if __name__ == "__main__":
    main()