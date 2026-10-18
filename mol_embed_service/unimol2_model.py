"""Native PyTorch Uni-Mol2 atom/pair encoder.

Copyright (c) DP Technology. Derived from Uni-Mol source revision
https://github.com/deepmodeling/Uni-Mol/tree/90f52c41299a1a582da0f9765e9f87aa21faa16a
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: this copyright notice
and this permission notice shall be included in all copies or substantial
portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .unimol2_config import CONFIGS


class Transition(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear_1 = nn.Linear(d, d)
        self.linear_2 = nn.Linear(d, d)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(x)))


class Attention(nn.Module):
    def __init__(self, d, p, h):
        super().__init__()
        self.num_heads = h
        self.linear_q = nn.Linear(d, d, bias=False)
        self.linear_k = nn.Linear(d, d, bias=False)
        self.linear_v = nn.Linear(d, d, bias=False)
        self.linear_o = nn.Linear(d, d)
        self.linear_bias = nn.Linear(p, h)
        self.norm = (d // h) ** -0.5

    def forward(self, x, pair, mask):
        b, n, d = x.shape
        h = self.num_heads
        dh = d // h
        q = (self.linear_q(x) * self.norm).view(b, n, h, dh).transpose(1, 2)
        k = self.linear_k(x).view(b, n, h, dh).transpose(1, 2)
        v = self.linear_v(x).view(b, n, h, dh).transpose(1, 2)
        bias = self.linear_bias(pair).permute(0, 3, 1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) + bias + mask, dim=-1)
        return self.linear_o((a @ v).transpose(1, 2).contiguous().view(b, n, d))


class OuterProduct(nn.Module):
    def __init__(self, d, p, h):
        super().__init__()
        self.linear_in = nn.Linear(d, 2 * h)
        self.linear_out = nn.Linear(h * h, p)

    def forward(self, x, mask, norm):
        a, b = self.linear_in(x).mul(mask).chunk(2, -1)
        # source ordering is [i,a,j,b] flattened
        z = (a[:, :, None, :, None] * b[:, None, :, None, :]).flatten(-2)
        return self.linear_out(z) * norm


class Triangle(nn.Module):
    def __init__(self, p, h):
        super().__init__()
        self.linear_ab_p = nn.Linear(p, 2 * h)
        self.linear_ab_g = nn.Linear(p, 2 * h)
        self.linear_g = nn.Linear(p, p)
        self.linear_z = nn.Linear(h, p)
        self.layer_norm_out = nn.LayerNorm(h)

    def forward(self, z, mask):
        g = self.linear_g(z)
        m = mask.unsqueeze(-1) * mask.shape[-2] ** -0.5
        a, b = (self.linear_ab_p(z) * m * torch.sigmoid(self.linear_ab_g(z))).chunk(2, -1)
        # (i,k,c) x (k,j,c) plus transposed orientation, matching source einsum.
        tri = torch.einsum("bikc,bjkc->bijc", a, b) + torch.einsum("bkic,bkjc->bijc", a, b)
        return g * self.linear_z(self.layer_norm_out(tri))


class EncoderLayer(nn.Module):
    def __init__(self, d, p, h, heads):
        super().__init__()
        self.self_attn = Attention(d, p, heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.ffn = Transition(d)
        self.final_layer_norm = nn.LayerNorm(d)
        self.x_layer_norm_opm = nn.LayerNorm(d)
        self.opm = OuterProduct(d, p, h)
        self.pair_layer_norm_ffn = nn.LayerNorm(p)
        self.pair_ffn = Transition(p)
        self.pair_layer_norm_trimul = nn.LayerNorm(p)
        self.pair_tri_mul = Triangle(p, h)

    def forward(self, x, z, attn, pair_mask, op_mask, op_norm):
        x = x + self.self_attn(self.self_attn_layer_norm(x), z, attn)
        x = x + self.ffn(self.final_layer_norm(x))
        z = z + self.opm(self.x_layer_norm_opm(x), op_mask, op_norm)
        z = z + self.pair_tri_mul(self.pair_layer_norm_trimul(z), pair_mask)
        z = z + self.pair_ffn(self.pair_layer_norm_ffn(z))
        return x, z


class Gaussian(nn.Module):
    def __init__(self, p, k):
        super().__init__()
        self.K = k
        mean = torch.linspace(0, 9, k)
        self.std = 1.0 * (mean[1] - mean[0])
        self.register_buffer("mean", mean)
        self.mul = nn.Embedding(512, 1, padding_idx=0)
        self.bias = nn.Embedding(512, 1, padding_idx=0)

    def forward(self, d, t):
        mul = self.mul(t).abs().sum(-2)
        bias = self.bias(t).sum(-2)
        x = (mul * d.unsqueeze(-1) + bias).expand(-1, -1, -1, self.K)
        return torch.exp(-0.5 * ((x.float() - self.mean.float()) / self.std) ** 2) / (
            (2 * 3.14159) ** 0.5 * self.std
        )


class NonLinear(nn.Module):
    def __init__(self, k, p):
        super().__init__()
        self.layer1 = nn.Linear(k, k)
        self.layer2 = nn.Linear(k, p)

    def forward(self, x):
        return self.layer2(F.gelu(self.layer1(x)))


class SE3(nn.Module):
    def __init__(self, p, k):
        super().__init__()
        self.gaussian = Gaussian(p, k)
        self.out_proj = NonLinear(k, p)

    def forward(self, d, t):
        return self.out_proj(self.gaussian(d, t))


class AtomFeature(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.atom_encoder = nn.Embedding(512, d, padding_idx=0)
        self.degree_encoder = nn.Embedding(128, d, padding_idx=0)
        self.vnode_encoder = nn.Embedding(1, d)

    def forward(self, data, tok):
        return torch.cat(
            (
                self.vnode_encoder.weight[None].expand(tok.shape[0], -1, -1),
                self.atom_encoder(data["atom_feat"]).sum(-2)
                + self.degree_encoder(data["degree"])
                + tok,
            ),
            1,
        )


class EdgeFeature(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.edge_encoder = nn.Embedding(64, p, padding_idx=0)
        self.shorest_path_encoder = nn.Embedding(512, p, padding_idx=0)
        self.vnode_virtual_distance = nn.Embedding(1, p)

    def forward(self, data):
        b, n = data["src_token"].shape
        out = torch.zeros(
            (b, n + 1, n + 1, self.edge_encoder.embedding_dim), device=data["src_token"].device
        )
        out[:, 1:, 1:] = self.shorest_path_encoder(data["shortest_path"]) + self.edge_encoder(
            data["edge_feat"]
        ).mean(-2)
        v = self.vnode_virtual_distance.weight
        out[:, 1:, 0] = v
        out[:, 0, :] = v
        return out


class UniMol2Model(nn.Module):
    def __init__(self, name="unimol2-84M", config=None):
        super().__init__()
        c = config or CONFIGS[name]
        d, p, h, k = c.width, c.pair_width, c.pair_hidden, c.kernels
        self.config = c
        self.embed_tokens = nn.Embedding(128, d, padding_idx=0)
        self.atom_feature = AtomFeature(d)
        self.edge_feature = EdgeFeature(p)
        self.se3_invariant_kernel = SE3(p, k)
        self.encoder = nn.Module()
        self.encoder.layer_norm = nn.LayerNorm(d)
        self.encoder.pair_layer_norm = nn.LayerNorm(p)
        self.encoder.layers = nn.ModuleList(
            [EncoderLayer(d, p, h, c.heads) for _ in range(c.layers)]
        )

    def forward(self, data, return_pair=False):
        x = self.atom_feature(data, self.embed_tokens(data["src_token"]))
        pair = self.edge_feature(data)
        n = data["atom_mask"]
        atom_mask = torch.cat(
            (torch.ones((n.shape[0], 1), device=n.device, dtype=n.dtype), n), 1
        ).float()
        pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
        pos = torch.cat((torch.zeros((n.shape[0], 1, 3), device=n.device), data["src_pos"]), 1)
        dist = torch.linalg.vector_norm(pos[:, :, None] - pos[:, None, :], dim=-1)
        types = torch.zeros((*dist.shape, 2), dtype=torch.long, device=dist.device)
        types[:, 1:, 1:] = data["pair_type"]
        pair = pair.clone()
        pair[:, 1:, 1:] += self.se3_invariant_kernel(dist, types)[:, 1:, 1:]
        attn = data["attn_bias"].to(x.dtype).unsqueeze(1).expand(-1, self.config.heads, -1, -1)
        x = self.encoder.layer_norm(x)
        pair = self.encoder.pair_layer_norm(pair)
        op = atom_mask.unsqueeze(-1) * atom_mask.shape[-1] ** -0.5
        norm = 1 / (1e-3 + torch.einsum("...bc,...dc->...bdc", op, op))
        for layer in self.encoder.layers:
            x, pair = layer(x, pair, attn, pair_mask, op, norm)
        return (x, pair) if return_pair else x[:, 0, :]
