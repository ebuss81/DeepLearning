import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Simple 1D CNN for time-series classification/regression.

    Input: (B, L, d_input) – internally transposed to (B, C, L) for Conv1d.
    """

    def __init__(self, d_input=1, d_output=3, start_dim=32, n_layers=2, dropout=0.0, kernel_size = 5, n_mlp=2):
        super().__init__()

        layers = []
        in_ch = d_input

        for i in range(n_layers):
            #k =max(5, kernel_sizes[0]kernel_sizes[0] - int(kernel_sizes[0]*0.1 *i)) # Note: Decrease kernel size with depth, min 5, self-define. Proabaly not needed due to effective receptive field growing with depth.
            raw_k = int(round(kernel_size * (0.8 ** i)))
            k = max(5, raw_k)
            # ensure kernel is odd
            if k % 2 == 0:
                k += 1

            #print(k)
            # Conv block: Conv -> BN -> ReLU
            d_model = 2 ** (i + start_dim)
            d_model = min(d_model, 1024)  # cap model size to 512 channels
            print(d_model)
            layers += [
                nn.Conv1d(in_ch, d_model, kernel_size=k, padding=kernel_size // 2),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),

            ]

            # Optional dropout
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            # Downsample every second block
            if i % 2 == 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2)) # other pooling parameteers?

            in_ch = d_model

        self.backbone = nn.Sequential(*layers)
        mlp_layers = []
        in_dim = d_model
        #n_mlp = 3

        mlp_layers.append(nn.AdaptiveAvgPool1d(1))
        mlp_layers.append(nn.Flatten())
        #print("hi", n_mlp)

        for h in range(n_mlp-1):
            #print(in_dim)
            out_dim = max(in_dim // 2, d_output)
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                mlp_layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim
        mlp_layers.append(nn.Linear(in_dim, d_output))
        self.head = nn.Sequential(*mlp_layers)



        """
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, C, L') -> (B, C, 1)
            nn.Flatten(),             # (B, C, 1) -> (B, C)
            nn.Linear(d_model, d_output),
        )
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_input)
        x = x.transpose(1, 2)   # -> (B, d_input, L)
        x = self.backbone(x)    # -> (B, d_model, L')
        x = self.head(x)        # -> (B, d_output)
        return x


def build_cnn1d(d_input, d_output, start_dim, n_layers, dropout, kernel_size, n_mlp=2):
    return CNN1D(
        d_input=d_input,
        d_output=d_output,
        start_dim=start_dim,
        n_layers=n_layers,
        dropout=dropout,
        kernel_size=kernel_size,
        n_mlp=n_mlp,
    )
