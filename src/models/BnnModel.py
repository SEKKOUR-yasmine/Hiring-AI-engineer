import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F


class BnnLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(BnnLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters for the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(torch.Tensor(out_features, in_features))

        # Parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_std = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.weight_std, -5.0)

        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_std, -5.0)

    def forward(self, x):
        weight = Normal(self.weight_mu, torch.exp(self.weight_std)).rsample()
        bias = Normal(self.bias_mu, torch.exp(self.bias_std)).rsample()
        return F.linear(x, weight, bias)


class BayesianModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianModel, self).__init__()
        self.layer1 = BnnLayer(input_size, hidden_size)
        self.layer2 = BnnLayer(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
