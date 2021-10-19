import torch
import torch.nn as nn
import torch.nn.functional as F

import modules


class Encoder(nn.Module):
    def __init__(
        self,
        n_mels,
        n_mega_blocks,
        n_sub_blocks,
        channels,
        mega_block_kernel_size,
        prolog_kernel_size=3,
        epilog_kernel_size=1,
        dropout=0.5,
    ):
        super(Encoder, self).__init__()
        self.prolog = modules.ConvBlock1d(n_mels, channels, prolog_kernel_size)
        self.mega_blocks = nn.Sequential(
            *[
                MegaBlock(
                    channels,
                    channels,
                    mega_block_kernel_size,
                    n_sub_blocks,
                    dropout=dropout,
                )
                for _ in range(n_mega_blocks)
            ]
        )
        self.epilog = modules.ConvBlock1d(channels, channels, epilog_kernel_size)

    def forward(self, x):
        x = self.prolog(x)
        x = self.mega_blocks(x)
        return self.epilog(x)


class MegaBlock(nn.Module):
    def __init__(self, input_size, channels, kernel_size, n_sub_blocks, dropout=0.5):
        super(MegaBlock, self).__init__()

        self.dropout = dropout

        channels = [input_size] + [channels] * n_sub_blocks
        self.sub_blocks = nn.Sequential(
            *[
                modules.ConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    activation="relu",
                    dropout=dropout,
                    depthwise=True,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ],
            modules.SqueezeExcitation()
        )

        self.skip_connection = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        y = self.skip_connection(x) + self.sub_blocks(x)
        return F.dropout(F.relu(y), p=self.dropout, training=self.training)
