import torch

from .model import Model
from .loss import NLL,Cross_entropy
from .loss_plt import loss_plt

from pathlib import Path

def save_model(model, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path)

def total_variation(theta):
    site_sum = theta.sum(dim=1)         
    tv = torch.abs(site_sum[1:] - site_sum[:-1]).sum()
    return tv

def train(model, X, y, save_path, lr=1e-3, epochs=500, Loss = NLL, tv_reg = True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_ls = []
    lambda_l1 = 1e-4 
    lambda_tv = 1e-3

    print("Training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = Cross_entropy(y_hat, y)

        l1_penalty = torch.sum(torch.abs(model.theta))
        
        if tv_reg:
            tv = lambda_tv * total_variation(model.theta)
            loss = loss + lambda_l1 * l1_penalty + tv
        else:
            loss = loss + lambda_l1 * l1_penalty
        
        loss.backward()
        optimizer.step()

        loss_ls.append(loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    save_model(model, save_path)
    loss_plt(loss_ls)
