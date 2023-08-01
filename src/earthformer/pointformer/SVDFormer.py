# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import torch
from torch import nn, einsum
from einops import repeat, rearrange
from math import sqrt

# classes

class SVDTransformer(nn.Module):
    def __init__(
            self,
            *,
            in_dim,
            hid_dim,
            out_dim,
            # heads=4,
            dim_head=64,
            mode=32,
    ):
        super().__init__()
        # self.heads = heads
        self.mode = mode
        # inner_dim = dim_head * heads

        self.mlp_q = nn.Linear(in_dim, hid_dim)
        self.mlp_k = nn.Linear(in_dim, hid_dim)
        self.mlp_v = nn.Linear(in_dim, hid_dim)

        self.mlp_d = nn.Linear(2, 1)
        self.mlp_out = nn.Linear(hid_dim, out_dim)

        # self.num_neighbors = num_neighbors
        #
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_out = nn.Linear(inner_dim, dim)
        #
        # self.pos_mlp = nn.Sequential(
        #     nn.Linear(3, pos_mlp_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(pos_mlp_hidden_dim, inner_dim)
        # )
        #
        # attn_inner_dim = inner_dim * attn_mlp_hidden_mult
        #
        # self.attn_mlp = nn.Sequential(
        #     nn.Conv2d(inner_dim, attn_inner_dim, 1, groups=heads),
        #     nn.ReLU(),
        #     nn.Conv2d(attn_inner_dim, inner_dim, 1, groups=heads),
        # )

    def attn(self, q, k, v):
        B, L1, m1 = q.shape
        B, L2, m2 = k.shape
        sim = einsum('b i m, b i n -> b m n', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b m n, b i n -> b i m', attn, v)
        return out

    def forward(self, q, k, v, mask=None):
        B, T, H, W, C = q.shape
        B, K, H, W, C = q.shape
        q = self.mlp_q(q)
        k = self.mlp_q(k)
        v = self.mlp_q(v)

        B, T, H, W, D = q.shape
        q, k, v = map(lambda t: rearrange(t, 'b t h w c -> b (h w) (t c)'), (q, k, v))

        Uq, dq, Vq = torch.linalg.svd(torch.nan_to_num(q, nan=0.0))
        Uk, dk, Vk = torch.linalg.svd(torch.nan_to_num(k, nan=0.0))
        Uv, dv, Vv = torch.linalg.svd(torch.nan_to_num(v, nan=0.0))

        mode = min(self.mode, H*W, min(T, K)*D)
        Uout = self.attn(Uq[..., :mode], Uk[..., :mode], Uv[..., :mode])
        assert Uout.shape == Uq[..., :mode].shape
        dout = self.mlp_d(torch.stack([dq[..., :mode], dk[..., :mode]], dim=-1)).squeeze(-1)
        assert dout.shape == dq[..., :mode].shape
        Vout = self.attn(Vq[..., :mode, :],
                         Vk[..., :mode, :],
                         Vv[..., :mode, :])
        assert Vout.shape == Vq[..., :mode, :].shape
        out = torch.einsum('b l m, b m -> b l m', Uout, dout)
        out = torch.einsum('b l m, b m n -> b l n', out, Vout)

        out = rearrange(out, 'b (h w) (t c) -> b t h w c', h=H, t=T)
        out = self.mlp_out(out)

        return out


class GlobalSVD(nn.Module):
    def __init__(
            self,
            *,
            H=32,
            W=32,
            dim=32,
            mode=8
    ):
        super().__init__()
        # self.heads = heads
        self.w = nn.Parameter(torch.randn(dim, H, W))
        self.mode = mode

    def forward(self, x):
        B, T, H, W, C = x.shape
        mode = self.mode
        x = rearrange(x, 'b t h w c -> b t c h w')

        Uq, dq, Vq = torch.linalg.svd(x)
        Uw, dw, Vw = torch.linalg.svd(self.w)

        U = torch.einsum('b t c h m, c h m -> b t c h m', Uq[..., :mode], Uw[..., :mode])
        d = torch.einsum('b t c m, c m -> b t c m', dq[..., :mode], dw[..., :mode])
        V = torch.einsum('b t c m w, c m w -> b t c m w', Vq[..., :mode, :], Vw[..., :mode, :])

        out = torch.einsum('b t c h m, b t c m -> b t c h m', U, d)
        out = torch.einsum('b t c h m, b t c m w -> b t h w c', out, V)

        return out


class GlobalConv(nn.Module):
    def __init__(
            self,
            *,
            H=32,
            W=32,
            dim=32,
            mode=8,
            T=10
    ):
        super().__init__()
        # self.heads = heads
        self.wr = nn.Parameter(torch.randn(T, dim, H, W))
        self.wi = nn.Parameter(torch.randn(T, dim, H, W))
        self.mode = mode

    def forward(self, x):
        B, T, H, W, C = x.shape
        mode = self.mode
        x = rearrange(x, 'b t h w c -> b t c h w')

        x_ = torch.fft.fft2(x, dim=(-2, -1))
        x_r = x_.real * self.wr
        x_i = x_.imag * self.wi
        x_out = torch.view_as_complex(torch.stack([x_r, x_i], dim=-1))
        out = torch.fft.ifft2(x_out, dim=(-2, -1)).real
        out = rearrange(out, 'b t c h w -> b t h w c')

        return out


class FullAttention(nn.Module):
    def __init__(self, q_length, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.mlp = nn.Parameter(torch.randn(q_length, q_length))

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, E = queries.shape
        _, S, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("ble,bse->bls", queries, keys)

        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bls,bsd->bld", A, values)

        V = torch.einsum("bld,pl->bpd", V, self.mlp)
        return V


if __name__ == '__main__':
    model = SVDTransformer(in_dim=32, hid_dim=32, out_dim=32)
    q = torch.randn(1, 13, 8, 8, 32)
    k = torch.randn(1, 10, 8, 8, 32)
    out = model.forward(q, k, k)
    a = 1
