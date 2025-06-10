# -*- coding: utf-8 -*-
import os
import sys

import torch

from ml_utils.utils.register import Register

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LambdaLayer(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


@Register(group_name="model", func_name="Identity")
class Identity(torch.nn.Identity):
    def forward(self, x, *args, **kwargs):
        return x


def main():
    pass


if __name__ == "__main__":
    main()
