import numpy as np
import torch

def dcn(x):
    return x.detach().cpu().numpy()

def to_torch_tensor(x, datatype='float'):
    if type(x) is torch.Tensor:
        pass
    elif type(x) is np.ndarray:
        if x.dtype == object:
            x = np.vstack(x)
        x = torch.from_numpy(x)
    elif type(x) is list:
        x = torch.Tensor(x)
    else:
        x = torch.Tensor([x])
    if datatype == 'float':
        x = x.float()
    elif datatype == 'long':
        x = x.long()
    elif datatype == 'int':
        x = x.int()
    else:
        raise Exception("type error")
    return x

def to_list(x):
    '''
    all data to list
    '''
    if type(x) is np.ndarray:
        x = x.tolist()
    elif type(x) is torch.Tensor:
        x = x.numpy().tolist()
    return x

def to_numpy(x):
    if type(x) is list:
        x = np.array(x)
    elif type(x) is torch.Tensor:
        x = x.detach().numpy()
    return x

def to_tf_tensor(x):
    pass
