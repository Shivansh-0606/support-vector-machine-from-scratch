"""
Support Vector Machine from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Demo scaffold: train a linear SVM from scratch on synthetic 2D data."""

import numpy as np


def make_toy_dataset(n_per_class=50):
    rng = np.random.default_rng(0)
    x_pos = rng.normal(loc=(2.0, 2.0), scale=0.8, size=(n_per_class, 2))
    x_neg = rng.normal(loc=(-2.0, -2.0), scale=0.8, size=(n_per_class, 2))
    x = np.vstack([x_pos, x_neg])
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    np.random.seed(0)

    # 1. Data prep
    x_raw, y = make_toy_dataset(n_per_class=60)
    x = standardize_features(x_raw)
    print("Data shapes: x =", x.shape, " y =", y.shape)
    print("First standardized rows:\n", np.round(x[:3], 3))
    print("Feature means ~0:", np.round(x.mean(axis=0), 3),
          " std ~1:", np.round(x.std(axis=0), 3))

    # 2. Forward pass with freshly initialized params
    n_features = x.shape[1]
    init_params = initialize_parameters(n_features)
    print("\nInitial params:", init_params)

    init_scores = compute_scores(x[:5], init_params)
    print("Initial scores (first 5):", np.round(init_scores, 4))
    print("Initial predictions:", predict_from_scores(init_scores))
    print("Hinge loss on example 0:",
          round(float(hinge_loss_example(float(init_scores[0]), float(y[0]))), 4))

    reg_lambda = 0.01
    init_obj = svm_objective(x, y, init_params, reg_lambda)
    print("Initial SVM objective:", round(float(init_obj), 4))

    # 3. A single manual gradient/update step (sanity check)
    grads = compute_gradients(x, y, init_params, reg_lambda)
    stepped_params = apply_update(init_params, grads, learning_rate=0.1)
    stepped_obj = svm_objective(x, y, stepped_params, reg_lambda)
    print("Objective after one manual step:", round(float(stepped_obj), 4))

    # 4. Full training loop
    trained_params = train_svm(
        x, y,
        learning_rate=0.05,
        reg_lambda=reg_lambda,
        n_epochs=200,
    )
    print("\nTrained params:", trained_params)
    print("Final SVM objective:",
          round(float(svm_objective(x, y, trained_params, reg_lambda)), 4))

    # 5. Predict & evaluate
    y_pred = predict_labels(x, trained_params)
    acc = float(np.mean(np.asarray(y_pred) == np.asarray(y)))
    print("Training accuracy:", round(float(acc), 4))
    print("First 10 predictions:", y_pred[:10].astype(int))
    print("First 10 true labels:", y[:10].astype(int))


if __name__ == "__main__":
    main()
