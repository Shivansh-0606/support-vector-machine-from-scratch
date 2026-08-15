"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    # TODO: rescale each column of x to have mean 0 and std 1 (leave zero-std columns alone).
    mean = np.mean(x , axis = 0)

    std = np.std(x , axis = 0)

    std_safe = np.where(std == 0, 1 ,std)

    return (x-mean) / std_safe

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM

    w = np.zeros(n_features)
    b = 0.0

    return {
        'w':w,
        'b':b
    }

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    return x @ params['w'] + params['b']

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    # arr = []

    # for i in scores:
    #     if i >= 0:
    #         arr.append(1)
    #     else:
    #         arr.append(-1)
    
    # return np.array(arr)

    return np.where(scores >=0 , 1 , -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    
    # hinge_loss = abs(y - score)

    # if hinge_loss == 1:
    #     hinge_loss = 0
    
    # return hinge_loss

    hinge_losses = np.maximum(0.0, 1.0 - y * score)

    return hinge_losses

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    scores = compute_scores(x,params)

    hinge_loss = hinge_loss_example(scores,y)

    w = params['w']
    reg_term = reg_lambda * (w @ w)

    ans = np.mean(hinge_loss) + reg_term

    return float(ans)

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # TODO: compute the gradient of the SVM objective wrt params['w'] and params['b'].
    scores = compute_scores(x , params)
    m = 1 - y*scores
    n = len(x)

    mask = (m > 0).astype(float)

    dw =  -((x.T @ (mask * y) / n)) + 2 * reg_lambda * params['w']

    db = -(np.sum(mask * y) / n)

    return {
        'dw':dw,
        'db':db
    }

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    
    new_w = params['w'] - learning_rate * grads['dw']

    new_b = params['b'] - learning_rate * grads['db']

    return {
        'w':new_w,
        'b':new_b
    }

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    # TODO: fit a linear SVM by repeatedly updating parameters over n_epochs passes.

    params =  initialize_parameters(x.shape[1])

    for epoch in range(n_epochs):
        grads =  compute_gradients(x, y, params, reg_lambda)

        params = apply_update(params , grads , learning_rate)
    
    return params

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

