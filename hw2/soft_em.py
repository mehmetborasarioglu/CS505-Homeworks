import os
import dill as pickle  
from models.modernizer import Modernizer
from from_file import load_parallel
from eval.cer import cer

MODEL_FILENAME = "trained_soft_em.pkl"

def train():
    if os.path.exists(MODEL_FILENAME):
        print(f"Loading pretrained model from {MODEL_FILENAME}...")
        with open(MODEL_FILENAME, "rb") as f:
            mod = pickle.load(f)
    else:
        print("Pretrained model not found. Training from scratch...")
        train_s_data, train_w_data = load_parallel("data/train.new", "data/train.old")
        test_s_data, test_w_data = load_parallel("data/test.new", "data/test.old")

        mod = Modernizer()
        mod.train_language_model(train_s_data)
        mod.init_typo_model(train_s_data, train_w_data)

        mod.flexible_train(train_s_data, train_w_data,
                           test_data=(test_s_data, test_w_data),
                           delta=10,
                           max_iters = 5,
                           converge_error=1e-12)

        with open(MODEL_FILENAME, "wb") as f:
            pickle.dump(mod, f)
        print(f"Model saved to {MODEL_FILENAME}.")

    test_s_data, test_w_data = load_parallel("data/test.new", "data/test.old")
    print("First 10 decodings from test set:")
    predictions = []
    for i, (decoded_str, logprob) in enumerate(mod.decode(test_w_data)):
        if i < 10:
            print(f"{decoded_str}\t{logprob}")
        predictions.append(decoded_str)

    score = cer(zip(test_s_data, predictions))
    print(f"Character Error Rate (CER): {score:.2%}")

if __name__ == "__main__":
    train()
