import torch
from .model import Model
from .encoding import one_hot_encode

def load_model(weight_path, seq_length, alphabet, device="cpu"):
    ALPHABETS = {
    "DNA": "ACGT",
    "RNA": "ACGU",
    "PROTEIN": "ACDEFGHIKLMNPQRSTVWY*"
    }
    alphabet_size = len(ALPHABETS[alphabet])
    state = torch.load(weight_path, map_location=device)
    if "theta" in state:
        expected_len, expected_alpha = state["theta"].shape
        if seq_length != expected_len:
            raise ValueError(
                f"Sequence length {seq_length} does not match model length {expected_len}"
            )
        if alphabet_size != expected_alpha:
            raise ValueError(
                f"Alphabet size {alphabet_size} does not match model alphabet size {expected_alpha}"
            )
    model = Model(seq_length, alphabet)
    model.load_state_dict(state)
    model.eval()
    return model

def predict(weight_path, seq, alphabet):
    model = load_model(weight_path, len(seq), alphabet)
    oh = one_hot_encode(seq, alphabet)
    X = torch.tensor(oh, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model(X)

