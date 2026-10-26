import numpy as np


def bce_gradient(
    input: np.ndarray,
    p: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    input:  shape (batch_size, in_features)
    p:      shape (batch_size, 1), sigmoid(linear(input, weight, bias))
    target: shape (batch_size, 1)

    Returns:
        grad_weight: shape (1, in_features)
        grad_bias: shape (1,)
    """
    # TODO: Implement the gradient formulas derived in Theory.
    # Do not use an autograd library -- these are the manual derivatives.
    n_samples= input.shape[0]
    error=p-target
    grad_weight= (error.T@input)/n_samples
    grad_bias=error.sum(axis=0)/n_samples
    return grad_weight,grad_bias
    
    pass
