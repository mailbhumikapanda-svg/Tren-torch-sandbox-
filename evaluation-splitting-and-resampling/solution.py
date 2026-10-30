import numpy as np


def train_test_split(
    input: np.ndarray, labels: np.ndarray, test_size: float, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    test_size: fraction of samples (0 to 1) to hold out as the test set.

    Returns:
        (train_input, test_input, train_labels, test_labels)
    """
    # TODO: Shuffle sample indices once (rng.permutation), take the
    # first round(n_samples * test_size) as the test set, the rest as
    # train.
    rng = np.random.default_rng(seed)
    n_samples = input.shape[0]
    shuffled_idx = rng.permutation(n_samples)

    n_test = int(round(n_samples * test_size))
    test_idx = shuffled_idx[:n_test]
    train_idx = shuffled_idx[n_test:]

    return input[train_idx], input[test_idx], labels[train_idx], labels[test_idx]
    pass


def k_fold_split(
    n_samples: int, k: int, seed: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
        a list of k (train_idx, val_idx) pairs. Each val_idx is one of
        k roughly-equal, non-overlapping folds of shuffled indices
        0..n_samples-1; the matching train_idx is every OTHER fold's
        indices concatenated together. Every sample appears in exactly
        one val_idx across the k pairs.
    """
    # TODO: Shuffle indices once, np.array_split into k folds. For each
    # fold i: val_idx = folds[i], train_idx = every other fold
    # concatenated.
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(n_samples)
    folds = np.array_split(shuffled_idx, k)

    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits
    pass
