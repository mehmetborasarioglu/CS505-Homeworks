# hard_em.py

from models.modernizer import Modernizer
from from_file import load_parallel
import matplotlib.pyplot as plt
from eval.cer import cer

def train() -> Modernizer:
    # Load training and test data
    train_s_data, train_w_data = load_parallel("data/train.new", "data/train.old")
    test_s_data, test_w_data = load_parallel("data/test.new", "data/test.old")

    # Initialize model
    mod = Modernizer()
    mod.train_language_model(train_s_data)
    mod.init_typo_model(train_s_data, train_w_data)

    # Log-likelihood tracking
    loglikelihoods = []

    # Run training
    mod.brittle_train(train_s_data, train_w_data,
                      test_data=(test_s_data, test_w_data),
                      delta=10,
                      max_iters=10,
                      converge_error=1e-5,
                      loglikelihoods=loglikelihoods)

    # Plot log-likelihood over iterations
    plt.plot(loglikelihoods, marker='o')
    plt.title("Log-Likelihood During Brittle EM Training")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.savefig("log_likelihood_plot.png")
    plt.show()

    # Evaluate final CER on test set
    test_preds = [out for out, _ in mod.decode(test_w_data)]
    score = cer(zip(test_s_data, test_preds))
    print(f"\nFinal CER: {score:.4f}")

    return mod

if __name__ == "__main__":
    train()
