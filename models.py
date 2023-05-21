import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torchinfo import summary

# =========================================================================
# The code references in this section are from https://github.com/microsoft/Swin-Transformer.
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # x = x.view(B, math.ceil(H / window_size), window_size, math.ceil(W / window_size), window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 不使用softmax, 则需要 QK^T/n, 故去掉self.scale
        # q = q * self.scale
        seq_len = q.size(-2)
        attn = (q @ k.transpose(-2, -1))
        attn = attn / seq_len

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,  in_dim, hidden_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.reduction = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.reduction = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm = norm_layer(in_dim)

    def forward(self, x):
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=nn.LayerNorm, stride=2, patch_padding=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        # x = self.proj(x)

        return x

class HTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C


        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# =========================================================================
class ReduceLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            HTransformerBlock(dim=dim, num_heads=num_heads,
                                 window_size=window_size, mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(in_dim=4*dim, hidden_dim=2*dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
                x1 = blk(x)

        if self.downsample is not None:
            x2 = self.downsample(x1)
        else:
            x2 = x1

        return x1, x2

class DecomposeLayer(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.dim = dim
        self.rebuildnet = nn.Linear(dim, 2*dim)

    def forward(self, x):
        B, H, W, C0 = x.shape
        C = int(C0/2)
        x_temp = self.rebuildnet(x)
        x_target = torch.zeros(B, H*2, W*2, self.dim//2, device=x.device)
        x_target[:, 0::2, 0::2, :] = x_temp[..., :C] # B H/2 W/2 C
        x_target[:, 1::2, 0::2, :] = x_temp[..., C:2*C]  # B H/2 W/2 C
        x_target[:, 0::2, 1::2, :] = x_temp[..., 2*C:3*C]  # B H/2 W/2 C
        x_target[:, 1::2, 1::2, :] = x_temp[..., 3*C:] # B H/2 W/2 C

        return x_target

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0., activation='gelu'):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GELU
        elif activation == 'tanh':
            act = nn.Tanh
        else:
            raise NameError('invalid activation')

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_scale=16):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.fourier_weight1 = nn.Parameter(
            torch.empty(in_channels, out_channels,
                        modes1, modes2, 2))
        self.fourier_weight2 = nn.Parameter(
            torch.empty(in_channels, out_channels,
                        modes1, modes2, 2))

        nn.init.xavier_uniform_(self.fourier_weight1, gain=1 / (in_channels * out_channels)
                                                           * np.sqrt((in_channels + out_channels) / init_scale))
        nn.init.xavier_uniform_(self.fourier_weight2, gain=1 / (in_channels * out_channels)
                                                           * np.sqrt((in_channels + out_channels) / init_scale))

    # Complex multiplication
    # def compl_mul2d(self, input, weights):
    #     # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    #     return torch.einsum("bixy,ioxy->boxy", input, weights)
    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),  # a[..., 0], b[..., 0]都是实部，a[..., 1], b[..., 1]都是虚部
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 2, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.fourier_weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.fourier_weight2)
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class Decodermap(nn.Module):
    def __init__(self, modes=12, width=64, num_spectral_layers=5, mlp_hidden_dim=128, output_dim=1, activation='gelu', padding=5, resolution=None, init_scale=16, add_pos=False):
        super().__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.add_pos = add_pos
        if add_pos:
            self.width = width + 2
        else:
            self.width = width
        self.num_spectral_layers = num_spectral_layers
        self.padding = padding  # pad the domain if input is non-periodic

        self.Spectral_Conv_List = nn.ModuleList([])

        for _ in range(num_spectral_layers):
            self.Spectral_Conv_List.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2, init_scale))

        self.Conv2d_list = nn.ModuleList([])

        for _ in range(num_spectral_layers):
            self.Conv2d_list.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1))

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise NameError('invalid activation')

        self.mlp = FeedForward(self.width, mlp_hidden_dim, output_dim)
        self.resolution = resolution

    def forward(self, x, pos=None):
        if self.add_pos:
            x = torch.cat((x, pos), dim=-1)

        x = x.permute(0, 3, 1, 2)
        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])
            # 矩阵上下左右侧扩充
            # 矩阵右侧和下侧扩充padding列行， 值全0.  x.size() = (batch_size,width,s+padding,s+padding)
            # 此时x为初始迭代值

        x1 = self.Spectral_Conv_List[0](x)
        x2 = self.Conv2d_list[0](x)
        x = self.act(x1 + x2)

        for i in range(1, self.num_spectral_layers - 1):
            x1 = self.Spectral_Conv_List[i](x)
            x2 = self.Conv2d_list[i](x)
            x = x1 + x2
            x = self.act(x)

        x1 = self.Spectral_Conv_List[-1](x)
        x2 = self.Conv2d_list[-1](x)
        # x = x1 + x2
        x = x1 + x2

        if self.padding:
            x = x[..., :self.resolution, :self.resolution]
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        return x

class HAttention(nn.Module):

    def __init__(self, in_chans=96, depths=[2, 2], num_heads=[4, 4], window_size=[8, 8], mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_layers = len(depths)
        self.num_features = int(in_chans)
        self.mlp_ratio = mlp_ratio

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.relayers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ReduceLayer(dim=int(self.num_features*2**i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
        for i_layer in range(self.num_layers):
            layer = DecomposeLayer(int(self.num_features*2**i_layer))
            self.relayers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        list_x_out = []
        x_downsample = x
        for layer in self.layers:
            x_att, x_downsample = layer(x_downsample)  # do attention
            list_x_out.append(x_att)

        for i_att in range(self.num_layers - 1, 0, -1):
            x_temp = self.relayers[i_att](list_x_out[i_att])
            list_x_out[i_att - 1] = list_x_out[i_att - 1] + x_temp

        x = list_x_out[0]


        return x

# --------------------------------------------------------------------------------
# Code of the model

class HANO2d(nn.Module):
    '''
        input shape: [batch, 1, res, res]
        output shape: [batch, res, res ,1]
    '''
    def __init__(self, R_dic):
        super(HANO2d, self).__init__()
        self.boundary_condition = R_dic['boundary_condition']
        self.feature_dim = R_dic['feature_dim']
        self.dropout = 0.0

        if 'in_dim' not in R_dic:
            R_dic['in_dim'] = 1

        # self.grid = R_dic['grid']
        # self.addpos = R_dic['addpos'] if 'addpos' in R_dic else 0

        self.resatt = R_dic['res_att']

        self.encoder = PatchEmbed(patch_size=R_dic['patch_size'], in_chans=R_dic['in_dim'],
                                  embed_dim=self.feature_dim, stride=R_dic['subsample_attn'],
                                  patch_padding=R_dic['patch_padding'])

        self.attn = HAttention(in_chans=self.feature_dim, depths=R_dic['depths'],
                               num_heads=R_dic['num_heads'], window_size=R_dic['window_size'])

        self.decoder = Decodermap(modes=R_dic['F_modes'], width=R_dic['F_width'], num_spectral_layers=R_dic['num_spectral_layers'],
                                  mlp_hidden_dim=R_dic['mlp_hidden_dim'], padding=R_dic['F_padding'], resolution=R_dic['res_output'])

        self.apply(self._init_weights)
        if 'y_norm' in R_dic:
            self.y_norm = R_dic['y_norm']
        else:
            self.y_norm = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.encoder(x)
        B, L, C = x.shape
        x = x.view(B, self.resatt, self.resatt, C)

        x = self.attn(x)
        # x.shape = [b,res_attn, res_attn,feature]

        x = self.decoder(x)
        # x.shape = [b,res_out,res_out,1]

        if self.y_norm is not None:
            x = torch.squeeze(x, dim=-1)
            x = self.y_norm.decode(x)
            x = torch.unsqueeze(x, dim=-1)

        if self.boundary_condition == 'dirichlet':
            x = x[:, 1:-1, 1:-1].contiguous()
            x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)

        return x

if __name__ == "__main__":
    R_dic = {}
    R_dic['seed'] = 0
    R_dic['model'] = 'HANO'
    R_dic['boundary_condition'] = 'dirichlet'
    R_dic['xGN'] = True
    R_dic['subsample_nodes'] = 1
    R_dic['subsample_attn'] = 2
    R_dic['patch_padding'] = 1
    R_dic['patch_size'] = 4
    R_dic['res_input'] = 512
    R_dic['res_att'] = 256
    R_dic['res_output'] = 256

    R_dic['feature_dim'] = 64  # feature dim, in order to enhance expressiveness
    R_dic['window_size'] = [4, 4, 4]
    R_dic['depths'] = [1, 1, 1]
    R_dic['num_heads'] = [1, 1, 1]

    R_dic['F_modes'] = 12  # modes of FNO
    R_dic['F_width'] = 64
    R_dic['num_spectral_layers'] = 5
    R_dic['mlp_hidden_dim'] = 128
    R_dic['F_padding'] = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HANO2d(R_dic)
    summary(model, input_size=(4, 1, 512, 512))
