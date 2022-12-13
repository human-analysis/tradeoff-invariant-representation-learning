# classification.py

from torch import nn

# __all__ = ['Classification']


# REVIEW: does this have to inherit nn.Module?
class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.weighted_total_loss = 0

        self.num_samples = 0

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        self.weighted_total_loss += loss * len(inputs)
        self.num_samples += len(inputs)
        return loss

    def compute(self):
        return self.weighted_total_loss / self.num_samples

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()
        self.softmax = nn.Softmax

    def __call__(self, inputs):
        inputs = self.softmax(inputs)
        loss = -(inputs * inputs.log()).sum(dim=1)
        return loss


class logLoss(nn.Module):
    def __init__(self):
        super(logLoss, self).__init__()
        self.weighted_total_loss = 0
        self.num_samples = 0

    def __call__(self, loss, batch_size):
        self.weighted_total_loss += loss * batch_size
        self.num_samples += batch_size

    def compute(self):
        return self.weighted_total_loss / self.num_samples

    def reset(self):
        self.weighted_total_loss = 0
        self.num_samples = 0