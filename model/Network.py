import torch
from torch import nn
from einops.layers.torch import Reduce


class SpectralCalibration(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MLP(nn.Module):
    def __init__(
            self,
            dim,
            dropout=0.
    ):
        super().__init__()
        self.linear1 = nn.Conv2d(dim, dim, 1)
        self.linear2 = nn.Conv2d(dim, dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = x = self.norm1(x.transpose(1, 3)).transpose(1, 3)
        x = self.act(self.linear1(x))
        x = self.dropout(self.linear2(x))
        x = x + y
        return self.norm2(x.transpose(1, 3)).transpose(1, 3)


class SimHSIC(nn.Module):
    def __init__(
            self,
            patch,
            output_repr,
            channels=200,
            dropout=0.1,
            dims=128,
    ):
        super().__init__()
        self.patch = patch
        self.sc = SpectralCalibration(channels, dims)
        self.layers_trans = nn.ModuleList([])

        self.layers_trans.append(nn.ModuleList([
            MLP(dim=int(dims), dropout=dropout),
            nn.BatchNorm2d(dims),
            nn.GELU(),
            nn.Conv2d(dims, dims, 1),
        ]))

        self.mlp_head1 = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims),
            nn.Linear(dims, 2 * dims),
            nn.Linear(2 * dims, output_repr)
        )

        self.mlp_head2 = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims),
            nn.Linear(dims, 2 * dims),
            nn.Linear(2 * dims, output_repr)
        )

    def forward(self, x):
        b, n, d = x.shape
        x = x.contiguous().view(b, n, self.patch, self.patch)
        x = x.squeeze(dim=1)
        x = self.sc(x)

        for mlp, bn, relu, pw in self.layers_trans:
            y = x
            x = mlp(x)
            x = pw(x) + y
            x = bn(x)
            x = relu(x)
        return self.mlp_head1(x), self.mlp_head2(x)


class Network(nn.Module):
    def __init__(self, patch, d_model, out_dims, channel, class_num, dropout):
        super(Network, self).__init__()
        self.feature_encoder = SimHSIC(patch=patch,
                                       output_repr=out_dims,
                                       channels=channel,
                                       dropout=dropout,
                                       dims=d_model)
        self.classifier = nn.Linear(in_features=out_dims, out_features=class_num)

    def forward(self, x, mode='train'):  # x
        repr1, repr2 = self.feature_encoder(x)
        output = self.classifier(repr1)
        return output, repr1, repr2
