import torch


def tv(theta):
    site_sum = theta.sum(dim=1)         
    tv = torch.abs(site_sum[1:] - site_sum[:-1]).sum()
    return tv

def l1(theta):
    return torch.sum(torch.abs(theta))