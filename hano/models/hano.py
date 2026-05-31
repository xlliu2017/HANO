import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0, activation="gelu"):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "tanh":
            act = nn.Tanh
        elif activation == "silu":
            act = nn.SiLU
        else:
            raise NameError("invalid activation")

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_scale=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.fourier_weight1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.fourier_weight2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        if init_scale:
            nn.init.uniform_(self.fourier_weight1, a=-self.scale / init_scale, b=self.scale / init_scale)
            nn.init.uniform_(self.fourier_weight2, a=-self.scale / init_scale, b=self.scale / init_scale)

    @staticmethod
    def compl_mul2d(x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x, out_resolution=None):
        x_ft = torch.fft.rfft2(x, norm="forward")
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.fourier_weight1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.fourier_weight2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="forward")


class SpectralDecoder(nn.Module):
    def __init__(
        self,
        modes=12,
        width=32,
        num_spectral_layers=5,
        mlp_hidden_dim=128,
        output_dim=1,
        activation="gelu",
        padding=6,
        resolution=None,
        init_scale=2,
        add_pos=False,
        shortcut=False,
        normalizer=None,
    ):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width + 2 if add_pos else width
        self.num_spectral_layers = num_spectral_layers
        self.padding = padding
        self.resolution = resolution
        self.add_pos = add_pos
        self.shortcut = shortcut
        self.normalizer = normalizer

        self.mlp = FeedForward(self.width, mlp_hidden_dim, output_dim, activation=activation)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2, init_scale) for _ in range(num_spectral_layers)]
        )
        self.local_layers = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1) for _ in range(num_spectral_layers)]
        )

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

    def _get_grid(self, x):
        batch_size, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32, device=x.device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(batch_size, 1, size_y, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32, device=x.device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(batch_size, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x, out_resolution=None):
        if self.add_pos:
            x = torch.cat((x, self._get_grid(x)), dim=-1)

        x = x.permute(0, 3, 1, 2)
        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.spectral_layers[0](x, out_resolution=out_resolution)
        x2 = self.local_layers[0](x)
        x = self.act(x1 + x2)
        x_shortcut = x if self.shortcut else None

        for i in range(1, self.num_spectral_layers - 1):
            x = self.act(self.spectral_layers[i](x, out_resolution=out_resolution) + self.local_layers[i](x))

        x = self.spectral_layers[-1](x, out_resolution=out_resolution) + self.local_layers[-1](x)
        if x_shortcut is not None:
            x = x + x_shortcut

        target_resolution = out_resolution or self.resolution
        if self.padding:
            x = x[..., :target_resolution, :target_resolution]
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)

        if self.normalizer is not None:
            x = self.normalizer.decode(x.squeeze(-1)).unsqueeze(-1)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, stride=2, patch_padding=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            (img_size[0] - patch_size[0] + 2 * patch_padding) // stride + 1,
            (img_size[1] - patch_size[1] + 2 * patch_padding) // stride + 1,
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_padding)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PlainWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or num_heads ** -1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        batch_windows, tokens, channels = qkv.shape
        qkv = qkv.reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads // 3).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = self.softmax(attn + relative_position_bias.unsqueeze(0))
        return (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels // 3)


class HTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.attn = PlainWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x):
        height, width = self.input_resolution
        batch_size, _, channels = x.shape
        x = x.view(batch_size, height, width, channels)
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels // 3)
        x = window_reverse(attn_windows, self.window_size, height, width)
        x = x.view(batch_size, height * width, channels // 3)
        return x


class HBasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                HTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        return sum(block(x) for block in self.blocks)


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, qkv_dim=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.qkv_dim = qkv_dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.reconstruction = nn.Linear(2 * dim, 4 * dim)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, direction="down", uplevel_result1=None):
        height, width = self.input_resolution
        if direction == "down":
            batch_size, _, channels = x.shape
            if self.qkv_dim:
                x = x.view(batch_size, height, width, 3, channels // 3)
                x0 = x[:, 0::2, 0::2, ...]
                x1 = x[:, 1::2, 0::2, ...]
                x2 = x[:, 0::2, 1::2, ...]
                x3 = x[:, 1::2, 1::2, ...]
                x = torch.cat([x0, x1, x2, x3], dim=-1)
                x = x.view(batch_size, -1, 4 * channels // 3)
                x = self.norm(x)
                x = self.reduction(x)
                return x.view(batch_size, height // 2 * width // 2, channels * 2)

            x = x.view(batch_size, height, width, channels)
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], dim=-1)
            x = self.norm(x.view(batch_size, -1, 4 * channels))
            return self.reduction(x)

        batch_size, _, channels = x.shape
        x = self.reconstruction(x).view(batch_size, height // 2, width // 2, 2 * channels)
        new_channels = channels // 2
        x = x.view(batch_size, height // 2, width // 2, 2, 2, new_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, new_channels)
        return x + uplevel_result1.view(batch_size, height, width, self.dim)


class HTransformer(nn.Module):
    def __init__(
        self,
        Decoder_paras=None,
        img_size=512,
        patch_size=4,
        in_chans=1,
        embed_dim=64,
        depths=(1, 1, 1),
        num_heads=(1, 1, 1),
        window_size=(4, 4, 4),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        stride=2,
        patch_padding=1,
    ):
        super().__init__()
        Decoder_paras = dict(Decoder_paras or {})
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=3 * embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            stride=stride,
            patch_padding=patch_padding,
        )
        patches_resolution = self.patch_embed.patches_resolution

        self.layers = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                HBasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size[i_layer],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                )
            )
        for i_layer in range(self.num_layers - 1):
            self.downsamplers.append(
                PatchMerging(
                    input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                    dim=int(embed_dim * 2 ** i_layer),
                    norm_layer=norm_layer,
                    qkv_dim=int(embed_dim * 3 * 2 ** i_layer),
                )
            )

        self.norm = norm_layer(self.embed_dim)
        decoder_width = Decoder_paras.get("width", embed_dim)
        self.decoder_proj = nn.Linear(embed_dim, decoder_width) if decoder_width != embed_dim else nn.Identity()
        Decoder_paras["width"] = decoder_width
        self.decoder = SpectralDecoder(**Decoder_paras)
        self.output_resolution = Decoder_paras.get("resolution")
        self.attn_resolution = patches_resolution[0]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, out_resolution=None):
        x = self.patch_embed(x)
        y_list = [self.layers[0](x)]
        for i in range(1, self.num_layers):
            x = self.downsamplers[i - 1](x)
            y_list.append(self.layers[i](x))

        merged = list(y_list)
        for i in range(self.num_layers - 1, 0, -1):
            merged[i - 1] = self.downsamplers[i - 1](y_list[i], direction="up", uplevel_result1=y_list[i - 1])

        x = self.norm(merged[0])
        x = self.decoder_proj(x)
        batch_size, _, channels = x.shape
        x = x.view(batch_size, self.attn_resolution, self.attn_resolution, channels)
        return self.decoder(x, out_resolution=out_resolution or self.output_resolution)


class HANO2d(nn.Module):
    """Upstream vFMM HANO architecture exposed behind the repository HANO API."""

    def __init__(self, r_dic):
        super().__init__()
        self.boundary_condition = r_dic["boundary_condition"]
        self.feature_dim = r_dic["feature_dim"]
        in_dim = r_dic.get("in_dim", 1)
        decoder_width = r_dic.get("F_width", self.feature_dim)

        decoder_params = {
            "modes": r_dic["F_modes"],
            "width": decoder_width,
            "padding": r_dic["F_padding"],
            "num_spectral_layers": r_dic["num_spectral_layers"],
            "activation": r_dic.get("activation", "gelu"),
            "mlp_hidden_dim": r_dic["mlp_hidden_dim"],
            "init_scale": r_dic.get("init_scale", 16),
            "add_pos": r_dic.get("add_pos", False),
            "shortcut": r_dic.get("shortcut", False),
            "resolution": r_dic["res_output"],
        }

        depths = tuple(r_dic["depths"])
        num_heads = tuple(r_dic["num_heads"][: len(depths)])
        window_size = tuple(r_dic["window_size"][: len(depths)])

        self.model = HTransformer(
            Decoder_paras=decoder_params,
            img_size=r_dic["res_input"],
            patch_size=r_dic["patch_size"],
            in_chans=in_dim,
            embed_dim=self.feature_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=r_dic.get("mlp_ratio", 4.0),
            qkv_bias=r_dic.get("qkv_bias", False),
            qk_scale=r_dic.get("qk_scale"),
            patch_norm=r_dic.get("patch_norm", True),
            stride=r_dic["subsample_attn"],
            patch_padding=r_dic["patch_padding"],
        )
        self.y_norm = r_dic.get("y_norm")

    def forward(self, x):
        x = self.model(x, out_resolution=self.model.output_resolution)

        if self.y_norm is not None:
            x = self.y_norm.decode(x.squeeze(-1)).unsqueeze(-1)

        if self.boundary_condition == "dirichlet":
            x = x[:, 1:-1, 1:-1].contiguous()
            x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)

        return x


class HANO(HANO2d):
    """Compatibility alias for HANO implementations expecting `HANO`."""


def window_partition(x, window_size):
    batch_size, height, width, channels = x.shape
    x = x.view(batch_size, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)



def window_reverse(windows, window_size, height, width):
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
