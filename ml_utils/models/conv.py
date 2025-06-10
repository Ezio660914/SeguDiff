# -*- coding: utf-8 -*-
import os
import sys
import warnings

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _convert_conv_transpose_padding_args_from_keras_to_torch(
        kernel_size, stride, dilation_rate, padding, output_padding
):
    """Convert the padding arguments from Keras to the ones used by Torch.
    Torch starts with an output shape of `(input-1) * stride + kernel_size`,
    then removes `torch_padding` from both sides, and adds
    `torch_output_padding` on the right.
    Because in Torch the output_padding can only be added to the right,
    consistency with Tensorflow is not always possible. In particular this is
    the case when both the Torch padding and output_padding values are
    strictly positive.
    """
    assert padding.lower() in {"valid", "same"}
    original_kernel_size = kernel_size
    kernel_size = (kernel_size - 1) * dilation_rate + 1

    if padding.lower() == "valid":
        # If output_padding is None, we fill it so that the shape of the output
        # is `(i-1)*s + max(k, s)`
        output_padding = (
            max(kernel_size, stride) - kernel_size
            if output_padding is None
            else output_padding
        )
        torch_padding = 0
        torch_output_padding = output_padding

    else:
        # When output_padding is None, we want the shape of the output to be
        # `input * s`, otherwise we use the value provided.
        output_padding = (
            stride - kernel_size % 2
            if output_padding is None
            else output_padding
        )
        torch_padding = max(
            -((kernel_size % 2 - kernel_size + output_padding) // 2), 0
        )
        torch_output_padding = (
                2 * torch_padding + kernel_size % 2 - kernel_size + output_padding
        )

    if torch_padding > 0 and torch_output_padding > 0:
        warnings.warn(
            f"You might experience inconsistencies across backends when "
            f"calling conv transpose with kernel_size={original_kernel_size}, "
            f"stride={stride}, dilation_rate={dilation_rate}, "
            f"padding={padding}, output_padding={output_padding}."
        )

    if torch_output_padding >= stride:
        raise ValueError(
            f"The padding arguments (padding={padding}) and "
            f"output_padding={output_padding}) lead to a Torch "
            f"output_padding ({torch_output_padding}) that is greater than "
            f"strides ({stride}). This is not supported. You can change the "
            f"padding arguments, kernel or stride, or run on another backend. "
        )

    return torch_padding, torch_output_padding


def compute_conv_transpose_padding_args_for_torch(
        num_spatial_dims,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
):
    torch_paddings = []
    torch_output_paddings = []
    for i in range(num_spatial_dims):
        output_padding_i = (
            output_padding
            if output_padding is None or isinstance(output_padding, int)
            else output_padding[i]
        )
        strides_i = stride if isinstance(stride, int) else stride[i]
        dilation_rate_i = (
            dilation
            if isinstance(dilation, int)
            else dilation[i]
        )
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_transpose_padding_args_from_keras_to_torch(
            kernel_size=kernel_size[i],
            stride=strides_i,
            dilation_rate=dilation_rate_i,
            padding=padding,
            output_padding=output_padding_i,
        )
        torch_paddings.append(torch_padding)
        torch_output_paddings.append(torch_output_padding)

    return torch_paddings, torch_output_paddings


class ConvTransposeNd(nn.Module):
    def __init__(self, num_spatial_dims, in_channels, out_channels, kernel_size, stride=1, padding="valid", output_padding=None, dilation=1, channel_last=True, **kwargs):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * num_spatial_dims
        torch_padding, torch_output_padding = compute_conv_transpose_padding_args_for_torch(
            num_spatial_dims=num_spatial_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
        )
        if isinstance(dilation, int):
            dilation = (dilation,) * num_spatial_dims

        if num_spatial_dims == 1:
            self.conv_t = torch.nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=torch_padding,
                output_padding=torch_output_padding,
                dilation=dilation,
                **kwargs
            )
        elif num_spatial_dims == 2:
            self.conv_t = torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=torch_padding,
                output_padding=torch_output_padding,
                dilation=dilation,
                **kwargs
            )
        elif num_spatial_dims == 3:
            self.conv_t = torch.nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=torch_padding,
                output_padding=torch_output_padding,
                dilation=dilation,
                **kwargs
            )
        else:
            raise NotImplementedError
        self.channel_last = channel_last

    def forward(self, x):
        if self.channel_last:
            x = _transpose_spatial_inputs(x)
        x = self.conv_t(x)
        if self.channel_last:
            x = _transpose_spatial_inputs(x)
        return x


def _compute_padding_length(
        input_length, kernel_length, stride, dilation_rate=1
):
    """Compute padding length along one dimension with support
    for asymmetric padding."""
    effective_k_size = (kernel_length - 1) * dilation_rate + 1
    if stride == 1:
        # total padding is kernel_size - 1
        total_padding = effective_k_size - 1
    else:
        # calc. needed padding for case with stride involved
        output_size = (input_length + stride - 1) // stride
        total_padding = max(
            0, (output_size - 1) * stride + effective_k_size - input_length
        )

    # divide padding evenly, with extra pixel going at the end if needed
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    return (left_padding, right_padding)


def standardize_tuple(value, n, name, allow_zero=False):
    """Transforms non-negative/positive integer/integers into an integer tuple.

    Args:
        value: int or iterable of ints. The value to validate and convert.
        n: int. The size of the tuple to be returned.
        name: string. The name of the argument being validated, e.g. "strides"
            or "kernel_size". This is only used to format error messages.
        allow_zero: bool, defaults to `False`. A `ValueError` will raised
            if zero is received and this argument is `False`.

    Returns:
        A tuple of n integers.
    """
    error_msg = (
        f"The `{name}` argument must be a tuple of {n} integers. "
        f"Received {name}={value}"
    )

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (
                    f"including element {single_value} of "
                    f"type {type(single_value)}"
                )
                raise ValueError(error_msg)

    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = ">= 0"
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = "> 0"

    if unqualified_values:
        error_msg += (
            f", including values {unqualified_values}"
            f" that do not satisfy `value {req_msg}`"
        )
        raise ValueError(error_msg)

    return value_tuple


def _apply_same_padding(
        inputs, kernel_size, stride, dilation=1, padding_mode='constant'
):
    """Apply same padding to the input tensor.

    This function will evaluate if the padding value is compatible with torch
    functions. To avoid calling `pad()` as much as possible, which may cause
    performance or memory issues, when compatible, it does not apply the padding
    to the tensor, but returns the input tensor and the padding value to pass to
    the torch functions. If not compatible, it returns the padded tensor and 0
    as the padding value.

    Returns:
        tensor: A padded tensor or the inputs.
        padding: The padding value, ready to pass to the torch functions.
    """
    spatial_shape = inputs.shape[2:]
    num_spatial_dims = len(spatial_shape)
    padding = []

    dilation = standardize_tuple(
        dilation, num_spatial_dims, "dilation_rate"
    )

    for i in range(num_spatial_dims):
        pad = _compute_padding_length(
            spatial_shape[i], kernel_size[i], stride[i], dilation[i]
        )
        padding.append(pad)

    # else, need to pad manually
    flattened_padding = []
    for pad in reversed(padding):
        flattened_padding.extend(pad)

    return F.pad(inputs, pad=tuple(flattened_padding), mode=padding_mode)


def _transpose_spatial_inputs(inputs):
    """Transpose inputs from channels_last to channels_first format."""
    # Torch pooling does not support `channels_last` format, so
    # we need to transpose to `channels_first` format.
    ndim = inputs.ndim - 2
    if ndim == 1:  # 1D case
        return torch.permute(inputs, (0, 2, 1))
    elif ndim == 2:  # 2D case
        return torch.permute(inputs, (0, 3, 1, 2))
    elif ndim == 3:  # 3D case
        return torch.permute(inputs, (0, 4, 1, 2, 3))
    raise ValueError(
        "Inputs must have ndim=3, 4 or 5, "
        "corresponding to 1D, 2D and 3D inputs. "
        f"Received input shape: {inputs.shape}."
    )


class ConvNd(nn.Module):
    def __init__(self, num_spatial_dims, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, channel_last=True):
        super().__init__()
        if num_spatial_dims == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode, device, dtype)
        elif num_spatial_dims == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode, device, dtype)
        elif num_spatial_dims == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode, device, dtype)
        else:
            raise NotImplementedError
        self.padding = padding
        self.padding_mode = 'constant' if padding_mode == 'zeros' else padding_mode
        self.channel_last = channel_last

    def forward(self, x):
        if self.channel_last:
            x = _transpose_spatial_inputs(x)
        if self.padding == 'same':
            x = _apply_same_padding(x, self.conv.kernel_size, self.conv.stride, self.conv.dilation, self.padding_mode)
        x = self.conv(x)
        if self.channel_last:
            x = _transpose_spatial_inputs(x)
        return x
