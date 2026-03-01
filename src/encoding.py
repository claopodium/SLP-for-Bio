import numpy as np
import torch

ALPHABETS = {
    "DNA": "ACGT",
    "RNA": "ACGU",
    "PROTEIN": "ACDEFGHIKLMNPQRSTVWY*"
}

def one_hot_encode(seq, alphabet="PROTEIN"):
    letters = ALPHABETS[alphabet]
    L = len(seq)
    A = len(letters)

    encoding = np.zeros((L, A), dtype=np.float32)
    letter_to_index = {c: i for i, c in enumerate(letters)}

    for i, c in enumerate(seq):
        if c not in letter_to_index:
            raise ValueError(f"Invalid character {c}")
        encoding[i, letter_to_index[c]] = 1.0

    return encoding

def to_oh_tensor(df):
    X_ls = []
    for seq in df['x']:
        oh = one_hot_encode(seq)
        X_ls.append(oh)

    X = np.stack(X_ls)
    X = torch.tensor(X,dtype=torch.float32)
    Y = df['y'].values
    Y = torch.tensor(Y,dtype=torch.float32)

    return X,Y
