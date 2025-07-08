"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union


@dataclass
class ModelArgs:
    dataset: str
    num_nodes: int
    d_model: int
    n_hidden: int
    n_layer: int
    ssm_layers: int
    n_feat: int
    n_class: int
    dropout: float
    weight: float
    d_state: int = 16
    expand: int = 1
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # if self.vocab_size % self.pad_vocab_size_multiple != 0:
        #     self.vocab_size += (self.pad_vocab_size_multiple
        #                         - self.vocab_size % self.pad_vocab_size_multiple)




class myMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.lin = nn.ModuleList()
        self.lin.append(nn.Linear(args.n_feat, args.n_hidden,bias=True))
        for i in range(args.n_layer-1):
            self.lin.append(nn.Linear(args.n_hidden, args.n_hidden,bias=True))
        self.lin.append(nn.Linear(args.n_hidden,args.n_class,bias=True))
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.ssm_layers)])
        self.r_layers=nn.ModuleList([ResidualBlock(args) for _ in range(args.ssm_layers)])
        self.inter=nn.Linear(1,args.d_model,bias=True)
        # self.inter = nn.Sequential(nn.Linear(1, args.d_model, bias=True),
        #                            nn.ReLU(),
        #                            nn.Linear(args.d_model, args.d_model, bias=True))
        self.outer=nn.Sequential(nn.Linear(args.d_model*2,1,bias=True))
        self.u=None
        self.filter=None
        self.reset_parameters()


    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for layer in self.lin:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, feat, U, lam):
        lam = lam.unsqueeze(-1).unsqueeze(0)

        u=self.inter(lam)
        r_u=u.flip(1)

        for i,(layer,r_layer) in enumerate(zip(self.layers,self.r_layers)):
            u = layer(u)
            r_u = r_layer(r_u)
            if i<len(self.layers)-1:
                u,r_u=F.relu(u),F.relu(r_u)

        u=self.outer(torch.cat((u.squeeze(),r_u.squeeze()),dim=1)).T
        u=self.args.weight*u/u.abs().max()
        # u=u.clip(0,1e6)
        self.u=u

        if self.training:
            self.u=u.detach().cpu().clone()

        filter = torch.mm(U, torch.mm(torch.diag(u.squeeze()), U.T))
        x = feat
        for num,lin in enumerate(self.lin):
            if num!=0:
                x = F.dropout(x, p=self.args.dropout, training=self.training)
            x =lin(x)
            if num!=len(self.lin)-1:
                x = F.relu(x)
            else:
                x=torch.mm(filter,x)

        return x



from .Mamba import Mamba1 as Mb
class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer=Mb(d_model=args.d_model)
        self.norm = RMSNorm(args.d_model)


    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        # output = self.mixer(self.norm(x)) + x
        output = self.mixer(x)

        return output




