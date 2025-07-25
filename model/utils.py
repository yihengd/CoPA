from torch import nn


class FFN(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x
