from torch import nn


class USPSConvNet(nn.Module):
    def __init__(self, nc=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(nc, nc * 2, 4, 2, 1),  # 8*8 spatial
            nn.SiLU(),
            nn.Conv2d(nc * 2, nc * 2, 3, 1, 1),  # 8*8 spatial
            nn.SiLU(),
            nn.Conv2d(nc * 2, nc * 4, 4, 2, 1),  # 4*4 spatial
            nn.SiLU(),
            nn.Conv2d(nc * 4, nc * 4, 3, 1, 1),  # 4*4 spatial
            nn.SiLU(),
            nn.Conv2d(nc * 4, nc * 8, 4, 2, 1),  # 2*2 spatial
            nn.SiLU(),
            nn.Conv2d(nc * 8, nc * 8, 2, 1, 0),  # 1*1 spatial
            nn.SiLU(),
        )
        self.final_linear = nn.Linear(nc * 8, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 16, 16)
        out = self.net(input)
        out = out.squeeze()
        return self.final_linear(out).squeeze()


class MNISTConvNet(nn.Module):
    def __init__(self, nc=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, nc, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(nc, nc * 2, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(nc * 2, nc * 2, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(nc * 2, nc * 4, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(nc * 4, nc * 4, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(nc * 4, nc * 8, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(nc * 8, nc * 8, 3, 1, 0),
            nn.SiLU(),
        )
        self.out = nn.Linear(nc * 8, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = out.squeeze()
        return self.out(out).squeeze()


class MLP(nn.Module):

    def __init__(self, insize, outsize, hsize, fc_num_layers, drop_prob=0.0, batchnorm=False, activation=nn.SiLU):
        super().__init__()
        fc_layers = []
        for i in range(fc_num_layers):
            fc_layers.append(nn.Linear(insize if (i == 0) else hsize, hsize))
            if batchnorm: fc_layers.append(nn.BatchNorm1d(hsize))
            fc_layers.append(activation())
            fc_layers.append(nn.Dropout(drop_prob))
        fc_layers.append(nn.Linear(hsize, outsize))
        self.net = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.net(x)
