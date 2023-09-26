# Adapted from https://github.com/openai/jukebox

import numpy as np
import torch.nn as nn
from modules.resnet import Resnet1D


def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"


class UnitEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, padding=2, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, padding=2, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, padding=2, kernel_size=5)
        self.blstm = nn.GRU(hidden_dim, hidden_dim//2, num_layers=2, bidirectional=True, batch_first=True)
        self.relu = nn.ReLU()
    
    def forward(self, units):
        units_embedded = self.relu(self.conv1(units))
        units_embedded = self.relu(self.conv2(units_embedded))
        units_embedded = self.relu(self.conv3(units_embedded))
        units_embedded = units_embedded.permute(0, 2, 1)
        units_embedded, _ = self.blstm(units_embedded)
        units_embedded = units_embedded.permute(0, 2, 1)
        
        return units_embedded
    
class F0Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim//16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim//16, out_channels=hidden_dim//8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim//8, out_channels=hidden_dim//4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=hidden_dim//4, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.blstm = nn.GRU(hidden_dim, hidden_dim//2, num_layers=2, bidirectional=True, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, f0):
        f0 = self.relu(self.conv1(f0))
        f0 = self.relu(self.conv2(f0))
        f0 = self.relu(self.conv3(f0))
        f0 = self.relu(self.conv4(f0))
        f0 = f0.permute(0, 2, 1)
        f0, _ = self.blstm(f0)
        f0 = f0.permute(0, 2, 1)

        return f0

class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False):
        super().__init__()
        blocks = []
        if type(stride_t) is tuple or type(stride_t) is list:
            start = True
            for s_t, d_t in zip(stride_t, down_t):
                if s_t % 2 == 0:
                    filter_t, pad_t = s_t * 2, s_t // 2
                else:
                    filter_t, pad_t = s_t * 2 + 1, s_t // 2 + 1
                if d_t > 0:
                    for i in range(d_t):
                        block = nn.Sequential(
                            nn.Conv1d(input_emb_width if i == 0 and start else width, width, filter_t, s_t, pad_t),
                            Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale), )
                        blocks.append(block)
                        start = False
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            if down_t > 0:
                for i in range(down_t):
                    block = nn.Sequential(
                        nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                        Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale), )
                    blocks.append(block)
                block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False,
                 reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []

        if type(stride_t) is tuple or type(stride_t) is list:
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for k, (s_t, d_t) in enumerate(zip(stride_t, down_t)):
                if d_t > 0:
                    if s_t % 2 == 0:
                        filter_t, pad_t = s_t * 2, s_t // 2
                    else:
                        filter_t, pad_t = s_t * 2 + 1, s_t // 2 + 1
                    end = k == len(stride_t) - 1
                    for i in range(d_t):
                        block = nn.Sequential(
                            Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out,
                                     res_scale=res_scale, reverse_dilation=reverse_decoder_dilation,
                                     checkpoint_res=checkpoint_res),
                            nn.ConvTranspose1d(width, input_emb_width if i == (d_t - 1) and end else width, filter_t,
                                               s_t, pad_t))
                        blocks.append(block)
        else:
            if down_t > 0:
                filter_t, pad_t = stride_t * 2, stride_t // 2
                block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
                blocks.append(block)
                for i in range(down_t):
                    block = nn.Sequential(
                        Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out,
                                 res_scale=res_scale, reverse_dilation=reverse_decoder_dilation,
                                 checkpoint_res=checkpoint_res),
                        nn.ConvTranspose1d(width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t,
                                           pad_t))
                    blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(
            input_emb_width if level == 0 else output_emb_width, output_emb_width, down_t, stride_t,
            **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            if type(stride_t) is tuple or type(stride_t) is list:
                emb, T = self.output_emb_width, T // np.prod([s ** d for s, d in zip(stride_t, down_t)])
            else:
                emb, T = self.output_emb_width, T // (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs


class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: DecoderConvBock(output_emb_width, output_emb_width, down_t,
                                                                      stride_t, **block_kwargs)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            if type(stride_t) is tuple or type(stride_t) is list:
                emb, T = self.output_emb_width, T * np.prod([s ** d for s, d in zip(stride_t, down_t)])
            else:
                emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
