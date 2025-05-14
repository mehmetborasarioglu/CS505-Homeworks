import matplotlib.pyplot as plt
from from_file import load_parallel
from models.modernizer import Modernizer

def train():
    # Load data
    train_s_data, train_w_data = load_parallel("data/train.new", "data/train.old")
    test_s_data, test_w_data = load_parallel("data/test.new", "data/test.old")

    # Initialize Modernizer
    mod = Modernizer()
    mod.train_language_model(train_s_data)
    mod.init_typo_model(train_s_data, train_w_data)

    # Track log-likelihoods
    log_likelihoods = []

    # Train using soft-EM
    mod.flexible_train(
        train_s_data=train_s_data,
        train_w_data=train_w_data,
        test_data=(test_s_data, test_w_data),
        delta=0.01,  # smoothing
        max_iters=30,
        converge_error=1e-4,
        loglikelihoods=log_likelihoods,
    )

    # Plot log-likelihood
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, marker="o")
    plt.title("Log-Likelihood over Soft-EM Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("soft_em_loglikelihood.png")
    plt.show()

    return mod


if __name__ == "__main__":
    print("Training soft-EM model...")
    model = train()
