def count_params(model):
    '''Number of parameters, the capacity figure reported for every E2 model.'''
    return sum(param.numel() for param in model.parameters())
