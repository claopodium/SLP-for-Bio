import torch

from src.model.slp import Model
from src.model.reg import tv, l1
from .loss import NLL,Cross_entropy
from .loss_plt import loss_plt

from pathlib import Path

def save_model(model, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path)

def train(model, X, y, save_path, lr=1e-3, epochs=1, Loss = NLL, tv_reg = True, regression = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_ls = []
    lambda_l1 = 1e-4 
    lambda_tv = 1e-3

    y_min = torch.min(y)
    y_max = torch.max(y)

    if regression:
        y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    print("Training...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        y_hat = model(X)

        loss = Loss(y_hat, y)

        l1_penalty = l1(model.theta)
        
        if tv_reg:
            ltv = tv(model.theta)
            loss = loss + lambda_l1 * l1_penalty + lambda_tv * ltv
        else:
            loss = loss + lambda_l1 * l1_penalty
        
        loss.backward()
        optimizer.step()

        loss_ls.append(loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    if regression:
        torch.save({
            "model_state_dict": model.state_dict(),
            "y_min": y_min,
            "y_max": y_max
        }, save_path)
    else:
        save_model(model, save_path)
    loss_plt(loss_ls)
