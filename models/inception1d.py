import torch
from torch import nn
# from .quantization import Conv1dSamePadding
import pytorch_lightning as pl
from typing import cast, Union, List
import torch.nn.functional as F
import traceback


class InceptionModel(pl.LightningModule):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 use_residuals: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 1, accelerator="cuda", groups=1, lr =0.01) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes,
            'lr': lr
        }
        self.accelerator = accelerator

        #channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
        #                                                                  num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
                                                                     num_blocks))
        #kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))


        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks)
                             )
        #print(kernel_sizes)

        self.blocks = nn.Sequential()

        for i in range(num_blocks):
            if i == 0:
                in_channel_block = in_channels
            else:
                in_channel_block = out_channels * (2 ** i)
            out_channel_block = out_channels * (2 ** (i + 1))
            kernel_size = max(5, int(round(kernel_sizes * (0.8 ** i))))
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            #print(in_channel_block, out_channel_block, kernel_size)
            block = InceptionBlock(in_channels=in_channel_block, out_channels=out_channel_block,
                        residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                        kernel_size=kernel_size, bit=32, layer=num_blocks, accelerator=self.accelerator,
                                     groups=groups)

            self.blocks.add_module(f'InceptionBlock_{i+1}', block)
        """
        self.blocks = nn.Sequential(*[
         InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                        residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                        kernel_size=kernel_sizes[i], bit=32, layer=num_blocks, accelerator=self.accelerator,
                                     groups=groups) for i in range(num_blocks)
                 ])
        """
        """
        i = 0
        self.layer1 = InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                                     residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                                     kernel_size=kernel_sizes[i], bit=32, layer=3, accelerator=self.accelerator,
                                     groups=groups)
        i = 1
        self.layer2 = InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                                     residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                                     kernel_size=kernel_sizes[i], bit=32, layer=3, accelerator=self.accelerator,
                                     groups=groups)
        i = 2
        self.layer3 = InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                                     residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                                     kernel_size=kernel_sizes[i], bit=32, layer=3, accelerator=self.accelerator,
                                     groups=groups)
        """

        # a global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        #self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes).to(self.accelerator)
        self.linear = nn.Linear(in_features=out_channel_block, out_features=num_pred_classes).to(self.accelerator)

        self.loss_fn = nn.CrossEntropyLoss()
        self.test_outputs = []  # buffer for predictions and labels
        self.test_results = None

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = x.transpose(1, 2)  # .to(self.accelerator)  # (B, C, L)

        x = self.blocks(x).mean(dim=-1)  # the mean is the global average pooling
        return self.linear(x)
        """
        x = x.transpose(1, 2)  # .to(self.accelerator)  # (B, C, L)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = self.linear(f3.mean(dim=-1))

        return out #[f1, f2, f3], out
        """

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=-1)
        acc = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        # Save outputs for later aggregation
        self.test_outputs.append({"y_hat": y_hat.detach(), "y": y.clone().detach()})
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)  # default lr = 0.001

    def on_test_epoch_end(self):
        if not self.test_outputs:
            print("⚠️ No test outputs collected — check test_step return.")
            return
        # Aggregate saved outputs
        y_hats = torch.cat([o["y_hat"] for o in self.test_outputs], dim=0)
        ys = torch.cat([o["y"] for o in self.test_outputs], dim=0)

        acc = (y_hats.argmax(dim=1) == ys.argmax(dim=1)).float().mean()

        # Save to instance for later use
        self.test_results = {
            "test_acc": acc,
            "y_hat": y_hats.cpu(),
            "y": ys.cpu(),
        }

        # Clear buffer (important for multi-run trainers)
        self.test_outputs.clear()


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41, bit=None, layer=3, accelerator="cuda", groups=1) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.accelerator = accelerator
        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
                                                kernel_size=1, bias=False, bit=bit, accelerator=self.accelerator,
                                                groups=groups)
            # self.bottleneck = nn.Conv1d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, bias=False, padding='same')

        kernel_size_s = [kernel_size // (2 ** i) for i in range(layer)] # Note check with cnn implementationt
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * layer
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=stride, bias=False, bit=bit,
                              accelerator=self.accelerator, groups=groups)
            # nn.Conv1d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernel_size_s[i], stride=stride, bias=False, padding='same')
            for i in range(len(kernel_size_s))
        ], nn.BatchNorm1d(num_features=out_channels).to(self.accelerator),
                                         nn.ReLU().to(self.accelerator))

        # TODO not needed
        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False, bit=bit, accelerator=self.accelerator,
                                  groups=groups),
                # nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                #           stride=stride, bias=False, padding='same'),
                nn.BatchNorm1d(out_channels).to(self.accelerator),
                nn.ReLU().to(self.accelerator)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x).to(self.accelerator)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    # NOTE we assume bit == 32

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, bit=None, accelerator="cuda"):
        super(Conv1dSamePadding, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups,
                                                bias)

        self.bit = bit
        self.additive = False  # config.additive
        self.accelerator = accelerator
        self.level = 0  # config.std_dev
        # self.bias = bias
        self.groups = groups

        self.weight = nn.Parameter(torch.randn(self.weight.size(), device=self.accelerator, dtype=torch.float32))

    def forward(self, x):
        input = x  # .to(self.accelerator)

        kernel, dilation, stride = self.weight.size(2), self.dilation[0], self.stride[0]

        l_out = l_in = input.size(2)
        padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)

        if padding % 2 != 0:
            input = F.pad(input, [0, 1])

        # print(f"weight device {self.weight.device}")
        # print(f"weight type {type(self.weight.device)}")
        # if self.weight.device.type == "cpu":
        #     stack = "".join(traceback.format_stack())
        #     raise RuntimeError(f"Model is running on CPU!\nCall stack:\n{stack}")
        # , bias=self.bias
        return F.conv1d(input=input, weight=self.weight, stride=stride,
                        padding=padding // 2,
                        dilation=dilation, groups=self.groups)  # .to(self.accelerator)

        # return conv1d_same_padding(x, self.weight, self.bias, self.stride,
        #                            self.dilation, self.groups, self.accelerator)

# def conv1d_same_padding(input, weight, bias, stride, dilation, groups, accelerator):
#
#     weight = weight.to(accelerator)
#     # print(weight.reshape(-1))
#     # stride and dilation are expected to be tuples.
#

def build_inception1d(n_classes = 2, num_blocks=3, d_input = 1, out_channels =10 , bottleneck_channels = 32, kernel_sizes= 41, lr = 0.01) -> InceptionModel:
    return InceptionModel(num_blocks = num_blocks,
                          in_channels = d_input,
                          out_channels = out_channels, #[10 ,20 ,40], that was default
                          bottleneck_channels = bottleneck_channels,
                          kernel_sizes = kernel_sizes,
                          use_residuals = True,
                          num_pred_classes = n_classes,
                          lr = lr
                          )
