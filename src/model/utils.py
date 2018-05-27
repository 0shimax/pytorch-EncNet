import torch
from torch import nn


def calculate_l1_loss(output, target, lagrange_coef=0.0005):
    l1_crit = nn.L1Loss(size_average=False)  # SmoothL1Loss
    reg_loss = l1_crit(output.argmax(dim=1).float(), target.float())

    return lagrange_coef * reg_loss


def smooth_in(model):
    l_noise = []
    for i, p in enumerate(model.parameters()):
        noise = torch.FloatTensor(p.shape).uniform_(-1, 1)
        p.data -= noise
        l_noise.append(noise)
        # model.parameters()[i] = p
    return l_noise


def smooth_out(model, l_noise):
    for i, (p, noise) in enumerate(zip(model.parameters(), l_noise)):
        p.data += noise
        l_noise.append(noise)
        # model.parameters()[i] = p
    # return l_noise
