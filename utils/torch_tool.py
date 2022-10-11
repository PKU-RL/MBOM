import torch
from torch.nn.parameter import Parameter
import types

def soft_update(params, ratio):
    assert abs(sum(ratio) - 1.0) < 1e-5, "soft_update ratio sum is not 1.0"
    assert len(params) == len(ratio), "soft_update param length error"
    if isinstance(params[0], types.GeneratorType):
        params = [list(p) for p in params]

    target = [Parameter(torch.zeros_like(p)) for p in params[0]]
    for j in range(len(params[0])):
        for i in range(0, len(ratio)):
            target[j].data.copy_(target[j].data + params[i][j] * ratio[i])
    return target