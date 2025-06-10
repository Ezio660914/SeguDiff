# -*- coding: utf-8 -*-
import os
import sys
from typing import overload, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from ml_utils.utils.register import Register

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, n_heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * n_heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.n_heads), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=self.n_heads)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.n_heads)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    Transformer block performs cross attention for conditioned inputs
    """

    def __init__(self, dim, n_heads, dim_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, context_dim=None, n_heads=n_heads, dim_head=dim_head, dropout=dropout)  # self attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, n_heads=n_heads, dim_head=dim_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.context_dim = context_dim

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=None) + x
        x = self.attn2(self.norm2(x), context=context if self.context_dim is not None else None) + x
        x = self.ff(self.norm3(x)) + x
        return x


class ContextTransformer(nn.Module):
    """
    Transformer encoder for multiple contexts
    """

    def __init__(self, in_channels, n_heads, dim_head, context_dim=None, dropout=0., norm_groups=32):
        super().__init__()
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * dim_head
        self.norm = Normalize(in_channels, norm_groups)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(inner_dim, n_heads, dim_head, dropout=dropout, context_dim=d)
             for d in context_dim]
        )
        self.proj_out = nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c l -> b l c').contiguous()
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        x = rearrange(x, 'b l c -> b c l').contiguous()
        x = self.proj_out(x)
        return x + x_in


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb, *args, **kwargs):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        pass


class TimestepEmbedSequential(nn.ModuleList, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    @overload
    def __init__(self, *args: nn.Module) -> None:
        ...

    @overload
    def __init__(self, modules: Optional[Iterable[nn.Module]] = None):
        ...

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Iterable):
            super().__init__(args[0])
        else:
            super().__init__(args)

    def forward(self, x, emb, context=None, *args, **kwargs):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ContextTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param conv_resample: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, out_channels=None, conv_resample=False, dims=1):
        super().__init__()
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        self.use_conv = conv_resample or (channels != out_channels)
        self.dims = dims
        if self.use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param conv_resample: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, out_channels=None, conv_resample=False, dims=1):
        super().__init__()
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        self.use_conv = conv_resample
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if conv_resample:
            self.op = conv_nd(dims, self.channels, self.out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self, channels, emb_channels, out_channels=None, dropout=0, use_conv=False, use_scale_shift_norm=False, dims=1, up=False, down=False, norm_groups=32):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = default(out_channels, channels)
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            Normalize(channels, norm_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, conv_resample=False, dims=dims)
            self.x_upd = Upsample(channels, conv_resample=False, dims=dims)
        elif down:
            self.h_upd = Downsample(channels, conv_resample=False, dims=dims)
            self.x_upd = Downsample(channels, conv_resample=False, dims=dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channels, norm_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, *args, **kwargs):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out.unsqueeze(-1)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


@Register(group_name="model", func_name="UNet1d")
class UNet1d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            model_channels,
            channel_mult=(1, 2, 4, 8),
            context_dim=None,
            num_res_blocks=1,
            transformer_n_heads=1,
            time_embed_dim=None,
            time_embed_max_period=10000,
            dropout=0,
            use_scale_shift_norm=True,
            conv_resample=True,
            res_block_updown=True,
            channel_last=True,
            norm_groups=32
    ):
        super().__init__()
        self.channel_last = channel_last
        time_embed_dim = default(time_embed_dim, model_channels * 4)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels, time_embed_max_period),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv1d(in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, out_channels=mult * model_channels, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm, norm_groups=norm_groups)
                ]
                ch = mult * model_channels
                assert ch % transformer_n_heads == 0
                layers.append(
                    ContextTransformer(ch, transformer_n_heads, ch // transformer_n_heads, context_dim, dropout, norm_groups)
                )
                self.input_blocks.append(TimestepEmbedSequential(layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                if res_block_updown:
                    down = ResBlock(ch, time_embed_dim, out_ch, dropout, False, use_scale_shift_norm, 1, down=True, norm_groups=norm_groups)
                else:
                    down = Downsample(ch, out_ch, conv_resample=conv_resample)
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        down
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
        assert ch % transformer_n_heads == 0
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm, norm_groups=norm_groups),
            ContextTransformer(ch, transformer_n_heads, ch // transformer_n_heads, context_dim, dropout, norm_groups),
            ResBlock(ch, time_embed_dim, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm, norm_groups=norm_groups),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, out_channels=mult * model_channels, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm, norm_groups=norm_groups)
                ]
                ch = model_channels * mult
                assert ch % transformer_n_heads == 0
                layers.append(
                    ContextTransformer(ch, transformer_n_heads, ch // transformer_n_heads, context_dim, dropout, norm_groups)
                )
                if level and i == num_res_blocks:
                    out_ch = ch
                    if res_block_updown:
                        down = ResBlock(ch, time_embed_dim, out_ch, dropout, False, use_scale_shift_norm, 1, up=True, norm_groups=norm_groups)
                    else:
                        down = Upsample(ch, out_ch, conv_resample=conv_resample)
                    layers.append(
                        down
                    )
                    ch = out_ch
                self.output_blocks.append(TimestepEmbedSequential(layers))
        self.out = nn.Sequential(
            Normalize(ch, norm_groups),
            nn.SiLU(),
            conv_nd(1, model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, context=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.channel_last:
            x = rearrange(x, "b ... c -> b c ...")
        hs = []
        emb = self.time_embed(timesteps)

        for module in self.input_blocks:
            x = module(x, emb, context)
            hs.append(x)
        x = self.middle_block(x, emb, context)
        for module in self.output_blocks:
            x = torch.cat([x, hs.pop()], dim=1)
            x = module(x, emb, context)
        out = self.out(x)
        if self.channel_last:
            out = rearrange(out, "b c ... -> b ... c")
        return out


def main():
    from torchinfo import summary
    m = UNet1d(1, 5, 64, (1, 2, 4, 8), 1, 2, 2)
    x = torch.rand(1, 768, 1)
    t = torch.randint(1000, size=(1,))
    c = torch.rand(1, 768, 1)
    out = m(x, t, c)
    summary(m, input_data=(x, t, c), depth=5, device='cpu')
    print(out.shape)


if __name__ == "__main__":
    main()
