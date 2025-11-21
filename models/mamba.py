import torch
import torch.nn as nn
from mamba_ssm import Mamba as MambaBlock  # avoid name clash


class MambaModel(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=128,      # smaller than 256 for 5k samples
        dropout=0.3,
        lr=None,
        prenorm=True,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=4,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_model = d_model

        # Linear encoder: (B, L, d_input) -> (B, L, d_model)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack of Mamba blocks as residual layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ),
                "dropout": nn.Dropout(dropout),
            })
            for _ in range(n_layers)
        ])

        # Linear decoder: (B, d_model) -> (B, d_output)
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        x: (B, L, d_input)
        """
        # Encode input features
        x = self.encoder(x)  # (B, L, d_model)

        # Apply Mamba layers with pre-norm + residual
        for layer in self.layers:
            residual = x
            z = layer["norm"](x)     # pre-norm
            z = layer["mamba"](z)    # (B, L, d_model)
            z = layer["dropout"](z)
            x = residual + z         # residual

        # Pooling over sequence length
        x = x.mean(dim=1)            # (B, d_model)

        # Decode to output
        x = self.decoder(x)          # (B, d_output)
        return x


def build_mamba(
    d_input, d_output, lr,
    d_model=128,
    n_layers=4,
    d_state=16,
    d_conv=4,
    expand=2,
    dropout=0.3,
):
    return MambaModel(
        d_input=d_input,
        d_output=d_output,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        lr=lr,
        prenorm=True,
    )