from models.modernizer import Modernizer
from from_file import load_parallel
from eval.cer import cer

def decode():
    # Load data
    train_s_data, train_w_data = load_parallel("data/train.old", "data/train.new")
    test_s_data, test_w_data = load_parallel("data/test.old", "data/test.new")

    # Initialize Modernizer and models
    mod = Modernizer()
    mod.train_language_model(train_s_data)
    mod.init_typo_model(train_s_data, train_w_data)

    # Decode the first 10 examples from test_w_data
    print("First 10 decodings from test set:")
    decoded_seqs = []
    for i, (decoded_str, logprob) in enumerate(mod.decode(test_w_data)):
        if i < 10:
            print(f"{i+1:2d}. {decoded_str}\t(logprob={logprob:.4f})")
        decoded_seqs.append(decoded_str)

    # Evaluate CER (Character Error Rate)
    error = cer(zip(test_s_data, decoded_seqs))
    print(f"\nCharacter Error Rate (CER): {error:.4f}")

    return mod

if __name__ == "__main__":
    decode()
