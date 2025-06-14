import numpy as np
import torch


def length_align(X, seq_len):
    """
    Align the length of the sequences to the specified sequence length.
    
    Parameters:
    X (ndarray): Input sequences.
    seq_len (int): Desired sequence length.

    Returns:
    ndarray: Aligned sequences with the specified length.
    """
    if seq_len < X.shape[-1]:
        X = X[..., :seq_len]  # Truncate the sequence if seq_len is shorter than the sequence length
    if seq_len > X.shape[-1]:
        padding_num = seq_len - X.shape[-1]  # Calculate padding length
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)  # Pad the sequence with zeros
    return X


def load_data(data_path):
    """
        Load and process data from a specified path.

        Parameters:
        data_path (str): Path to the data file.
        seq_len (int): Desired sequence length.

        Returns:
        tuple: Processed feature tensor and label tensor.
    """
    # Load data from file
    data = np.load(data_path)

    # Extract and process data
    X = data["X"]
    # X = np.sign(X)
    # Align the length of sequences
    X = length_align(X, 5000)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)

    if 'y' in data:
        y = data["y"]
        y = torch.tensor(y, dtype=torch.long)
        return X, y
    return X


def load_iter(X, y=None, batch_size=100, is_train=True, num_workers=5):
    if y is not None:
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
    else:
        # The case of unlabeled data
        return torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
