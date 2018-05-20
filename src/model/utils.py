from torch import nn


def calculate_l1_loss(output, target, lagrange_coef=0.0005):
    l1_crit = nn.L1Loss(size_average=False)  # SmoothL1Loss
    reg_loss = l1_crit(output.argmax(dim=1).float(), target.float())

    return lagrange_coef * reg_loss
