import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_

from .components import (
    DecomposeLayer,
    FeedForward,
    PatchEmbed,
    PatchMerging,
    ReduceLayer,
)


class SpectralConv2d(nn.Module):
    """2D spectral convolution used in the HANO decoder."""

    def __init__(self, in_channels, out_channels, modes1, modes2, init_scale=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.fourier_weight1 = nn.Parameter(torch.empty(in_channels, out_channels, modes1, modes2, 2))
        self.fourier_weight2 = nn.Parameter(torch.empty(in_channels, out_channels, modes1, modes2, 2))

        nn.init.xavier_uniform_(
            self.fourier_weight1,
            gain=1 / (in_channels * out_channels) * np.sqrt((in_channels + out_channels) / init_scale),
        )
        nn.init.xavier_uniform_(
            self.fourier_weight2,
            gain=1 / (in_channels * out_channels) * np.sqrt((in_channels + out_channels) / init_scale),
        )

    @staticmethod
    def complex_matmul_2d(a, b):
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
        ], dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 2, device=x.device)

        out_ft[:, :, : self.modes1, : self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.fourier_weight1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.fourier_weight2
        )
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class Decodermap(nn.Module):
    """FNO-style decoder that maps latent features back to output fields."""

    def __init__(
        self,
        modes=12,
        width=64,
        num_spectral_layers=5,
        mlp_hidden_dim=128,
        output_dim=1,
        activation="gelu",
        padding=5,
        resolution=None,
        init_scale=16,
        add_pos=False,
    ):
        super().__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.add_pos = add_pos
        self.width = width + 2 if add_pos else width
        self.num_spectral_layers = num_spectral_layers
        self.padding = padding

        self.Spectral_Conv_List = nn.ModuleList()
        for _ in range(num_spectral_layers):
            self.Spectral_Conv_List.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2, init_scale))

        self.Conv2d_list = nn.ModuleList()
        for _ in range(num_spectral_layers):
            self.Conv2d_list.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1))

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise NameError("invalid activation")

        self.mlp = FeedForward(self.width, mlp_hidden_dim, output_dim)
        self.resolution = resolution

    def forward(self, x, pos=None):
        if self.add_pos:
            x = torch.cat((x, pos), dim=-1)

        x = x.permute(0, 3, 1, 2)
        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.Spectral_Conv_List[0](x)
        x2 = self.Conv2d_list[0](x)
        x = self.act(x1 + x2)

        for i in range(1, self.num_spectral_layers - 1):
            x1 = self.Spectral_Conv_List[i](x)
            x2 = self.Conv2d_list[i](x)
            x = self.act(x1 + x2)

        x1 = self.Spectral_Conv_List[-1](x)
        x2 = self.Conv2d_list[-1](x)
        x = x1 + x2

        if self.padding:
            x = x[..., : self.resolution, : self.resolution]
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        return x


class HAttention(nn.Module):
    """Hierarchical window-attention encoder for multiscale feature extraction."""

    def __init__(
        self,
        in_chans=96,
        depths=(2, 2),
        num_heads=(4, 4),
        window_size=(8, 8),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.num_features = int(in_chans)
        self.mlp_ratio = mlp_ratio

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.relayers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ReduceLayer(
                dim=int(self.num_features * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        for i_layer in range(self.num_layers):
            self.relayers.append(DecomposeLayer(int(self.num_features * 2 ** i_layer)))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        list_x_out = []
        x_downsample = x
        for layer in self.layers:
            x_att, x_downsample = layer(x_downsample)
            list_x_out.append(x_att)

        for i_att in range(self.num_layers - 1, 0, -1):
            x_temp = self.relayers[i_att](list_x_out[i_att])
            list_x_out[i_att - 1] = list_x_out[i_att - 1] + x_temp

        return list_x_out[0]


class HANO2d(nn.Module):
    """Hierarchical Attention Neural Operator for 2D PDE surrogate modeling."""

    def __init__(self, r_dic):
        super().__init__()
        self.boundary_condition = r_dic["boundary_condition"]
        self.feature_dim = r_dic["feature_dim"]

        if "in_dim" not in r_dic:
            r_dic["in_dim"] = 1

        self.resatt = r_dic["res_att"]

        self.encoder = PatchEmbed(
            patch_size=r_dic["patch_size"],
            in_chans=r_dic["in_dim"],
            embed_dim=self.feature_dim,
            stride=r_dic["subsample_attn"],
            patch_padding=r_dic["patch_padding"],
        )

        self.attn = HAttention(
            in_chans=self.feature_dim,
            depths=r_dic["depths"],
            num_heads=r_dic["num_heads"],
            window_size=r_dic["window_size"],
        )

        self.decoder = Decodermap(
            modes=r_dic["F_modes"],
            width=r_dic["F_width"],
            num_spectral_layers=r_dic["num_spectral_layers"],
            mlp_hidden_dim=r_dic["mlp_hidden_dim"],
            padding=r_dic["F_padding"],
            resolution=r_dic["res_output"],
        )

        self.apply(self._init_weights)
        self.y_norm = r_dic.get("y_norm")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.encoder(x)
        bsz, _, channels = x.shape
        x = x.view(bsz, self.resatt, self.resatt, channels)

        x = self.attn(x)
        x = self.decoder(x)

        if self.y_norm is not None:
            x = torch.squeeze(x, dim=-1)
            x = self.y_norm.decode(x)
            x = torch.unsqueeze(x, dim=-1)

        if self.boundary_condition == "dirichlet":
            x = x[:, 1:-1, 1:-1].contiguous()
            x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)

        return x
