import torch

class AverageMeter:
    '''
        Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val    = 0
        self.avg    = 0
        self.sum    = 0
        self.count  = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def compute_accuracy(output, target):
    '''
        output: (B, N) 
        target: (B)
    '''
    target = target.squeeze()
    _, pred = torch.max(output, dim=1)
    return torch.mean((pred == target).float())
