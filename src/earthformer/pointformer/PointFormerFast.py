# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

from natten import NeighborhoodAttention2D, NeighborhoodAttention1D

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import optim

try:
    from .Embed_PointFormer import DataEmbedding
    from .multihead_point_transformer_pytorch import MultiheadPointTransformerLayer
    from .SVDFormer import SVDTransformer, FullAttention, GlobalSVD, GlobalConv
except:
    from src.earthformer.pointformer.Embed_PointFormer import DataEmbedding
    from src.earthformer.pointformer.multihead_point_transformer_pytorch import MultiheadPointTransformerLayer
    from src.earthformer.pointformer.SVDFormer import SVDTransformer, FullAttention, GlobalSVD, GlobalConv
# from layers.Causal_Conv import CausalConv
# from layers.Multi_Correlation import AutoCorrelation, AutoCorrelationLayer, CrossCorrelation, CrossCorrelationLayer, \
#     MultiCorrelation
# from layers.Corrformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, \
#     my_Layernorm, series_decomp


from collections import namedtuple


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DownSampling3D(nn.Module):
    """The 3D down-sampling layer.

    3d_interp_2d:
        x --> conv3d_3X3X3 (mid_dim) + leaky_relu --> downsample (bilinear) --> conv2d_3x3
    2d_interp_2d:
        x --> conv2d_3x3 (mid_dim) + leaky_relu --> downsample (bilinear) --> conv2d_3x3

    We add an additional conv layer before the

    For any options, if the target_size is the same as the input size, we will skip the bilinear downsampling layer.
    """
    def __init__(self, original_size, target_size, in_channels, out_dim, mid_dim=16, act_type='leaky',
                 arch_type='2d_interp_2d'):
        """

        Parameters
        ----------
        original_size
            The original size of the tensor. It will be a tuple/list that contains T, H, W
        target_size
            Will be a tuple/list that contains T_new, H_new, W_new
        in_channels
            The input channels
        out_dim
            The output dimension of the layer
        mid_dim
            Dimension of the intermediate projection layer
        act_type
            Type of the activation
        arch_type
            Type of the layer.
        """
        super(DownSampling3D, self).__init__()
        self.arch_type = arch_type
        self.original_size = original_size
        self.target_size = target_size
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        if self.arch_type == '3d_interp_2d':
            self.inter_conv = nn.Conv3d(in_channels=in_channels, out_channels=mid_dim, kernel_size=(3, 3, 3),
                                        padding=(1, 1, 1))
            self.act = nn.LeakyReLU(0.1)
            # self.act = nn.Tanh()
        elif self.arch_type == '2d_interp_2d':
            self.inter_conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_dim, kernel_size=(3, 3),
                                        padding=(1, 1))
            self.act = nn.LeakyReLU(0.1)
            # self.act = nn.Tanh()
        else:
            raise NotImplementedError
        self.conv = nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=(3, 3), padding=(1, 1))
        # self.init_weights()
        # 添加 norm
        self.norm3d = nn.InstanceNorm3d(out_dim)

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (N, T, H, W, C)

        Returns
        -------
        out
            Shape (N, T_new, H_new, W_new, C_out)
        """
        B, T, H, W, C_in = x.shape
        if self.arch_type == '3d_interp_2d':
            x = self.act(self.inter_conv(x.permute(0, 4, 1, 2, 3)))  # Shape(B, mid_dim, T, H, W)
            if self.original_size[0] == self.target_size[0]:
                # Use 2D interpolation
                x = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(B * T, self.mid_dim, H, W), size=self.target_size[1:])  # Shape (B * T_new, mid_dim, H_new, W_new)
            else:
                # Use 3D interpolation
                x = F.interpolate(x, size=self.target_size)  # Shape (B, mid_dim, T_new, H_new, W_new)
                x = x.permute(0, 2, 1, 3, 4).reshape(B * self.target_size[0], self.mid_dim,
                                                     self.target_size[1], self.target_size[2])
        elif self.arch_type == '2d_interp_2d':
            x = self.act(self.inter_conv(x.permute(0, 1, 4, 2, 3).reshape(B * T, C_in, H, W)))  # (B * T, mid_dim, H, W)

            if self.original_size[0] == self.target_size[0]:
                # Use 2D interpolation
                x = F.interpolate(x, size=self.target_size[1:])  # Shape (B * T_new, mid_dim, H_new, W_new)
            else:
                # Use 3D interpolation
                x = F.interpolate(x.reshape(B, T, C_in, H, W).permute(0, 2, 1, 3, 4), size=self.target_size)  # Shape (B, mid_dim, T_new, H_new, W_new)
                x = x.permute(0, 2, 1, 3, 4).reshape(B * self.target_size[0], self.mid_dim,
                                                     self.target_size[1], self.target_size[2])
        else:
            raise NotImplementedError
        x = self.conv(x)  # Shape (B * T_new, out_dim, H_new, W_new)
        x = x.reshape(B, self.target_size[0], self.out_dim, self.target_size[1], self.target_size[2]) \
            .permute(0, 1, 3, 4, 2)
        x = self.norm3d(x)
        return x

class Upsample3DLayer(nn.Module):
    """Upsampling based on nn.UpSampling and Conv3x3.

    If the temporal dimension remains the same:
        x --> interpolation-2d (nearest) --> conv3x3(dim, out_dim)
    Else:
        x --> interpolation-3d (nearest) --> conv3x3x3(dim, out_dim)

    """
    def __init__(self,
                 dim,
                 out_dim,
                 target_size,
                 temporal_upsample=False,
                 kernel_size=3,
                 layout='THWC',
                 conv_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
        out_dim
        target_size
            Size of the output tensor. Will be a tuple/list that contains T_new, H_new, W_new
        temporal_upsample
            Whether the temporal axis will go through upsampling.
        kernel_size
            The kernel size of the Conv2D layer
        layout
            The layout of the inputs
        """
        super(Upsample3DLayer,self).__init__()
        self.conv_init_mode = conv_init_mode
        self.target_size = target_size
        self.out_dim = out_dim
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.up = nn.Upsample(size=target_size, mode='nearest')  # 3D upsampling
        else:
            self.up = nn.Upsample(size=(target_size[1], target_size[2]), mode='nearest')  # 2D upsampling
        self.conv = nn.Conv2d(in_channels=dim, out_channels=out_dim, kernel_size=(kernel_size, kernel_size),
                              padding=(kernel_size // 2, kernel_size // 2))
        assert layout in ['THWC', 'CTHW']
        self.layout = layout

        self.upnorm3d = nn.InstanceNorm3d(out_dim)

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C) or (B, C, T, H, W)

        Returns
        -------
        out
            Shape (B, T, H_new, W_out, C_out) or (B, C, T, H_out, W_out)
        """
        if self.layout == 'THWC':
            B, T, H, W, C = x.shape
            if self.temporal_upsample:
                x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
                out = self.conv(self.up(x)).permute(0, 2, 3, 4, 1)
            else:
                assert self.target_size[0] == T
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B * T, C, H, W)
                x = self.up(x)
                out = self.conv(x).permute(0, 2, 3, 1).reshape((B,) + self.target_size + (self.out_dim,))
        out = self.upnorm3d(out)
        return out
        # elif self.layout == 'CTHW':
        #     B, C, T, H, W = x.shape
        #     if self.temporal_upsample:
        #         return self.conv(self.up(x))
        #     else:
        #         assert self.output_size[0] == T
        #         x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        #         x = x.reshape(B * T, C, H, W)
        #         return self.conv(self.up(x)).reshape(B, self.target_size[0], self.out_dim, self.target_size[1],
        #                                              self.target_size[2]).permute(0, 2, 1, 3, 4)


class PointAttentionLayer(nn.Module):
    def __init__(self, *, H, W, T, in_dim, hid_dim, out_dim, device, configs, neighbor_r=10):
        super(PointAttentionLayer, self).__init__()
        self.configs = configs
        # self.point_transformer = MultiheadPointTransformerLayer(H=H, W=W, dim=in_dim, pos_mlp_hidden_dim=8,
        #                                                         attn_mlp_hidden_mult=4, neighbor_r=neighbor_r,
        #                                                         device=device)

        self.point_transformer = NeighborhoodAttention2D(dim=in_dim, kernel_size=7, num_heads=8)
        # self.svd_former = SVDTransformer(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)
        # self.global_svd = GlobalSVD(H=H, W=W, dim=in_dim, T=T)
        self.global_conv = GlobalConv(H=H, W=W, dim=in_dim, T=T)

        self.full_attention = FullAttention(q_length=T)

        self.norm_q = nn.LayerNorm(in_dim * 1)
        self.norm_kv = nn.LayerNorm(in_dim * 2)

    # def get_pos(self, x):
    #     B, T, H, W, C = x.shape
    #     position_H = repeat(torch.arange(0, H), 'h -> h w', w=W)
    #     position_W = repeat(torch.arange(0, W), 'w -> h w', h=H)
    #     position = torch.stack([position_H, position_W], dim=-1).float()
    #     position = repeat(position, 'h w c -> b t h w c', b=B, t=T)
    #     return position

    def forward(self, q):
        # q.Shape=(B, K, H, W, C)
        # k.Shape=(B, T, H, W, C)
        # v.Shape=(B, T, H, W, C)
        B, K, H, W, C = q.shape

        # pos = self.get_pos(q)

        # pos = rearrange(pos, 'b t h w c -> (b t) (h w) c')
        if self.configs.wPT:
            q = rearrange(q, 'b t h w c -> (b t) h w c')
            q = self.point_transformer(q)
            q = rearrange(q, '(b t) h w c -> b t h w c', b=B)

        # q, k, v = self.norm(torch.cat([q_out, k, v], dim=-1)).chunk(3, dim=-1)
        q = self.norm_q(q)
        # k, v = self.norm_kv(torch.cat([k, v], dim=-1)).chunk(2, dim=-1)

        # q = self.global_svd(q)
        if self.configs.wGC:
            q = self.global_conv(q) + q
        # out = self.svd_former(q, k, v)
        # out = rearrange(q, 'b t h w c -> (b h w) t c')

        # k_ = torch.cat([q, k], dim=-2)
        # v_ = torch.cat([q, v], dim=-2)
        # out = self.full_attention(q, k_, v_)
        # out = rearrange(out, '(b h w) t c -> b t h w c', b=B, h=H, w=W)
        return q


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        if isinstance(configs, dict):
            configs = namedtuple('Struct', configs.keys())(*configs.values())
        # device = configs.device
        device = DEVICE
        self.verbose = configs.verbose
        T = configs.seq_len
        self.T = T
        # self.label_len = configs.label_len
        K = configs.pred_len
        self.K = K
        # self.node_num = configs.node_num
        # self.node_list = configs.node_list  # node_num = node_list[0]*node_list[1]*node_list[2]...
        # self.output_attention = configs.output_attention
        H = configs.height
        self.H = H
        W = configs.width
        self.W = W
        self.c_in = configs.c_in
        D = configs.d_model
        self.d_model = D
        self.neighbor_r = configs.neighbor_r

        # embed
        self.enc_embedding = DataEmbedding(configs=configs, H=self.H, W=self.W, c_in=self.c_in, d_model=D)
        self.dec_embedding = DataEmbedding(configs=configs, H=self.H, W=self.W, c_in=self.c_in, d_model=D)

        self.downsample_divide2 = DownSampling3D(original_size=(T, H, W), target_size=(T, H//2, W//2),
                                                 in_channels=D, out_dim=D*2, mid_dim=D)

        self.downsample_divide4 = DownSampling3D(original_size=(T, H//2, W//2), target_size=(T, H//4, W//4),
                                                 in_channels=D*2, out_dim=D*4, mid_dim=D*2)

        self.downsample_divide8 = DownSampling3D(original_size=(T, H//4, W//4), target_size=(T, H//8, W//8),
                                                 in_channels=D*4, out_dim=D*8, mid_dim=D*4)

        self.downsample_dec_divide8 = DownSampling3D(original_size=(K, H, W), target_size=(K, H // 8, W // 8),
                                                     in_channels=D, out_dim=D*8, mid_dim=D*4)

        self.upsample_divide2 = Upsample3DLayer(dim=D*2, out_dim=D, target_size=(K, H, W))

        self.upsample_divide4 = Upsample3DLayer(dim=D*4, out_dim=D*2, target_size=(K, H//2, W//2))

        self.upsample_divide8 = Upsample3DLayer(dim=D*8, out_dim=D*4, target_size=(K, H//4, W//4))

        self.attn = PointAttentionLayer(H=H, W=W, T=T, in_dim=D, hid_dim=D, out_dim=D, device=device, configs=configs, neighbor_r=self.neighbor_r)
        self.attn_divide2 = PointAttentionLayer(H=H//2, W=W//2, T=T, in_dim=D*2, hid_dim=D*2, out_dim=D*2, device=device, configs=configs, neighbor_r=self.neighbor_r//1.4)
        self.attn_divide4 = PointAttentionLayer(H=H//4, W=W//4, T=T, in_dim=D*4, hid_dim=D*4, out_dim=D*4, device=device, configs=configs, neighbor_r=self.neighbor_r//2)
        self.attn_dec_divide8 = PointAttentionLayer(H=H//8, W=W//8, T=K+T, in_dim=D*8, hid_dim=D*8, out_dim=D*8, device=device, configs=configs, neighbor_r=self.neighbor_r//2.8)
        self.attn_dec_divide4 = PointAttentionLayer(H=H//4, W=W//4, T=K+T, in_dim=D*4, hid_dim=D*4, out_dim=D*4, device=device, configs=configs, neighbor_r=self.neighbor_r//2)
        self.attn_dec_divide2 = PointAttentionLayer(H=H//2, W=W//2, T=K+T, in_dim=D*2, hid_dim=D*2, out_dim=D*2, device=device, configs=configs, neighbor_r=self.neighbor_r//1.4)
        self.attn_dec = PointAttentionLayer(H=H, W=W, in_dim=D, T=K+T, hid_dim=D, out_dim=D, device=device, configs=configs, neighbor_r=self.neighbor_r)

        self.mlp_out = nn.Linear(D, configs.c_out)

    def forward(self, x_enc): #, x_mark_enc, x_dec, x_mark_dec,
                # enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B, T, H, W, C = x_enc.shape
        x_mark_enc = repeat(torch.arange(0, 10).unsqueeze(-1), 'l d -> b l d', b=B).to(x_enc.device)
        x_mark_dec = repeat(torch.arange(10, 20).unsqueeze(-1), 'l d -> b l d', b=B).to(x_enc.device)
        x_dec = repeat(x_enc.mean(dim=1), 'b h w d -> b t h w d', t=self.K).detach().to(x_enc.device)

        if self.verbose:
            a = x_enc
            g = self.enc_embedding.enc_embedding.token_embedding.embed.weight.grad
            print('x_enc status: mean={}, max={}, min={}'.format(a.mean(), a.max(), a.min()))
            if g is not None:
                print('enc_embedding grad: mean={}, max={}, min={}'.format(g.mean(), g.max(), g.min()))

        x = self.enc_embedding(x_enc, x_mark_enc)

        x = self.attn(x) + x
        x_divide2 = self.downsample_divide2(x)

        x_divide2 = self.attn_divide2(x_divide2) + x_divide2
        x_divide4 = self.downsample_divide4(x_divide2)

        x_divide4 = self.attn_divide4(x_divide4) + x_divide4
        x_divide8 = self.downsample_divide8(x_divide4)

        B, T, H8, W8, D8 = x_divide8.shape
        # x_dec = repeat(x_enc.clone().detach().mean(dim=1), 'b h w d -> b t h w d', t=self.pred_len)
        x_dec = self.dec_embedding(x_dec, x_mark_dec)
        x_dec_divide8 = self.downsample_dec_divide8(x_dec)
        B, T, H8, W8, C = x_dec_divide8.shape

        x_dec_divide8 = torch.cat([x_divide8, x_dec_divide8], dim=1)
        x_dec_divide8 = (self.attn_dec_divide8(x_dec_divide8) + x_dec_divide8)[:, -self.K:, ...]
        x_dec_divide4 = self.upsample_divide8(x_dec_divide8)

        x_dec_divide4 = torch.cat([x_divide4, x_dec_divide4], dim=1)
        x_dec_divide4 = (self.attn_dec_divide4(x_dec_divide4) + x_dec_divide4)[:, -self.K:, ...]
        x_dec_divide2 = self.upsample_divide4(x_dec_divide4)

        x_dec_divide2 = torch.cat([x_divide2, x_dec_divide2], dim=1)
        x_dec_divide2 = (self.attn_dec_divide2(x_dec_divide2) + x_dec_divide2)[:, -self.K:, ...]
        x_dec_divide1 = self.upsample_divide2(x_dec_divide2)

        x_dec_divide1 = torch.cat([x, x_dec_divide1], dim=1)
        x_dec_divide1 = (self.attn_dec(x_dec_divide1) + x_dec_divide1)[:, -self.K:, ...]

        out = x_dec_divide1
        out = self.mlp_out(out)
        if self.verbose:
            a = out
            g = self.mlp_out.weight.grad
            print('out status: mean={}, max={}, min={}'.format(a.mean(), a.max(), a.min()))
            if g is not None:
                print('out_mlp grad: mean={}, max={}, min={}'.format(g.mean(), g.max(), g.min()))

        return out


if __name__ == '__main__':
    class Configs(object):
        seq_len = 10
        pred_len = 10
        height = 32
        width = 32
        c_in = 3
        c_out = 3
        d_model = 32
        temporal_type = 'index'
        neighbor_r = 6
        device = torch.device('cpu')
        verbose = 0
        wPT = 1
        wGC = 1


    configs = Configs()
    model = Model(configs=configs)
    model_optim = optim.Adam(model.parameters(), lr=0.1)
    for i in range(0, 10):

    # Shape of the input tensor. It will be (T, H, W, C_in)
        B = 1
        input_shape = (configs.seq_len, configs.height, configs.width, 3)
        output_shape = (configs.pred_len, configs.height, configs.width, 3)
        # input_shape_mark = (configs.seq_len, 4)

        input_x = torch.randn(B, *input_shape)
        input_x[:, :, :, :5] = 0
        input_x_mark = torch.ones(B, configs.seq_len, 1).long()

        dec_x = torch.randn(B, *output_shape)
        dec_x_mark = torch.ones(B, configs.pred_len, 1).long()

        # out = model.forward(input_x, input_x_mark, dec_x, dec_x_mark)
        out = model.forward(input_x)
        # out = model.forward(input_x, input_x_mark, dec_x, dec_x_mark)

        target = torch.randn(out.shape)
        criterion = nn.MSELoss()
        loss = criterion(out, target)
        grad1 = model.enc_embedding.token_embedding.embed.weight.grad
        grad11 = model.mlp_out.weight.grad
        loss.backward()
        model_optim.step()
        grad2 = model.enc_embedding.token_embedding.embed.weight.grad
        grad22 = model.mlp_out.weight.grad
        print(grad2.mean())
        assert out.shape == dec_x.shape
        print(out.shape)
    a = 1
