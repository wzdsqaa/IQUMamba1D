import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd

from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from torch.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(int(dim))
        self.mamba = Mamba(
                d_model=int(dim),
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *dims)
        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        dims = x.shape[2:]
        x_flat = x.flatten(2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *dims)
        return out

    @autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)
        return out


class SkipConnectionProcessor(nn.Module):
    """
    跳跃连接特征处理模块，用于处理编码器特征并与解码器特征更好地融合
    包含通道注意力、特征重标定和噪声抑制机制
    """
    def __init__(self, 
                 skip_channels: int,
                 upsampled_channels: int,
                 conv_op: Type[_ConvNd],
                 norm_op: Type[nn.Module],
                 norm_op_kwargs: dict,
                 nonlin: Type[nn.Module],
                 nonlin_kwargs: dict,
                 reduction_ratio: int = 8):
        super().__init__()
        
        self.skip_channels = skip_channels
        self.upsampled_channels = upsampled_channels

        self.feature_align = nn.Sequential(
            conv_op(skip_channels, skip_channels, kernel_size=1),
            norm_op(skip_channels, **norm_op_kwargs),
            nonlin(**nonlin_kwargs)
        )

        self.channel_attention = ChannelAttention1D(skip_channels, reduction_ratio)

        self.adaptive_fusion = AdaptiveFusion1D(
            skip_channels, upsampled_channels, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs
        )

        self.feature_refine = nn.Sequential(
            conv_op(skip_channels, skip_channels, kernel_size=3, padding=1),
            norm_op(skip_channels, **norm_op_kwargs),
            nonlin(**nonlin_kwargs),
            conv_op(skip_channels, skip_channels, kernel_size=1),
            norm_op(skip_channels, **norm_op_kwargs)
        )

        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, skip_features, upsampled_features):

        identity = skip_features

        x = self.feature_align(skip_features)

        x = self.channel_attention(x)

        x = self.adaptive_fusion(x, upsampled_features)

        x = self.feature_refine(x)

        x = self.residual_weight * x + (1 - self.residual_weight) * identity
        
        return x


class ChannelAttention1D(nn.Module):
    """1D通道注意力模块"""
    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        hidden_channels = max(1, channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        

        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention


class AdaptiveFusion1D(nn.Module):
    """自适应特征融合模块"""
    def __init__(self, 
                 skip_channels: int,
                 upsampled_channels: int,
                 conv_op: Type[_ConvNd],
                 norm_op: Type[nn.Module],
                 norm_op_kwargs: dict,
                 nonlin: Type[nn.Module],
                 nonlin_kwargs: dict):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.weight_generator = nn.Sequential(
            conv_op(upsampled_channels, skip_channels // 4, 1),
            nonlin(**nonlin_kwargs),
            conv_op(skip_channels // 4, skip_channels, 1),
            nn.Sigmoid()
        )

        self.cross_interaction = nn.Sequential(
            conv_op(skip_channels + upsampled_channels, skip_channels, 1),
            norm_op(skip_channels, **norm_op_kwargs),
            nonlin(**nonlin_kwargs)
        )
        
    def forward(self, skip_features, upsampled_features):

        global_context = self.global_pool(upsampled_features)
        adaptive_weight = self.weight_generator(global_context)

        weighted_skip = skip_features * adaptive_weight
        
        if upsampled_features.size(-1) != skip_features.size(-1):
            upsampled_features_resized = F.interpolate(
                upsampled_features, 
                size=skip_features.size(-1), 
                mode='linear', 
                align_corners=False
            )
        else:
            upsampled_features_resized = upsampled_features

        combined = torch.cat([weighted_skip, upsampled_features_resized], dim=1)
        output = self.cross_interaction(combined)
        
        return output


class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        
        if self.conv3:
            x = self.conv3(x)
        
        y += x
        return self.act2(y)
    
class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]
        
        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i / j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        self.conv_pad_sizes = [[k//2 for k in ks] for ks in kernel_sizes]

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op = conv_op,
                input_channels = input_channels,
                output_channels = stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0][0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
            ), 
            *[BasicBlockD(
                conv_op=conv_op,
                input_channels=stem_channels,
                output_channels=stem_channels,
                kernel_size=kernel_sizes[0],
                stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ) for _ in range(n_blocks_per_stage[0] - 1)]
        )

        input_channels = stem_channels
        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s][0],
                    stride=strides[s][0],
                    use_1x1conv=True,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs,
                ),
                *[BasicBlockD(
                    conv_op=conv_op,
                    input_channels=features_per_stage[s],
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[s] - 1)]
            )

            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    MambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s]
                    )
                )                
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]

class UNetResDecoder(nn.Module):
    def __init__(self, encoder, num_classes, n_conv_per_stage: Union[int, List[int]], deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1) if isinstance(n_conv_per_stage, int) else n_conv_per_stage

        stages = []
        upsample_layers = []
        seg_layers = []
        skip_processors = []  
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s][0]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='linear' if encoder.conv_op == nn.Conv1d else 'nearest'
            ))

            skip_processors.append(
                SkipConnectionProcessor(
                    skip_channels=input_features_skip,
                    upsampled_channels=input_features_skip,
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs
                )
            )

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip if s < n_stages_encoder - 1 else input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    padding=encoder.conv_pad_sizes[-(s + 1)][0],
                    stride=1,
                    use_1x1conv=True,
                ),
                *[BasicBlockD(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    stride=1,
                    conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                ) for _ in range(n_conv_per_stage[s-1] - 1)]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.skip_processors = nn.ModuleList(skip_processors) 

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                processed_skip = self.skip_processors[s](skips[-(s+2)], x)
                x = torch.cat((x, processed_skip), 1)
            x = self.stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))
            lres_input = x
        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]

class IQUMamba1D(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: List[int],
                 conv_op: Type[nn.Conv1d],
                 kernel_sizes: List[int],
                 strides: List[int],
                 n_conv_per_stage: List[int],
                 num_classes: int,
                 n_conv_per_stage_decoder: List[int],
                 conv_bias: bool = True,
                 norm_op: Type[nn.Module] = nn.InstanceNorm1d,
                 norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
                 nonlin: Type[nn.Module] = nn.LeakyReLU,
                 nonlin_kwargs: dict = {'inplace': True},
                 deep_supervision: bool = False,
                 ):
        super().__init__()
        self.encoder = ResidualMambaEncoder(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes],
            strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)