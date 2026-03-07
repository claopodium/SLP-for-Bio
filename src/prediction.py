import torch
from src.model.slp import Model
from .encoding import one_hot_encode


def load_model(weight_path, seq_length, alphabet, device="cpu"):

    ALPHABETS = {
        "DNA": "ACGT",
        "RNA": "ACGU",
        "PROTEIN": "ACDEFGHIKLMNPQRSTVWY*"
    }

    alphabet_size = len(ALPHABETS[alphabet])

    checkpoint = torch.load(weight_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if "theta" in state_dict:
        expected_len, expected_alpha = state_dict["theta"].shape

        if seq_length != expected_len:
            raise ValueError(
                f"Sequence length {seq_length} does not match model length {expected_len}"
            )

        if alphabet_size != expected_alpha:
            raise ValueError(
                f"Alphabet size {alphabet_size} does not match model alphabet size {expected_alpha}"
            )

    model = Model(seq_length, alphabet)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint

def predict(weight_path, seq, alphabet, device="cpu"):

    model, checkpoint = load_model(weight_path, len(seq), alphabet, device)

    oh = one_hot_encode(seq, alphabet)
    X = torch.tensor(oh, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(X)

    if "y_min" in checkpoint and "y_max" in checkpoint:
        y_min = float(checkpoint["y_min"])
        y_max = float(checkpoint["y_max"])
        y_pred = (y_max - y_min) * y_pred + y_min

    return y_pred.item()
