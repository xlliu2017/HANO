"""HANO model implemented with multigrid-style attention blocks.

This module replaces the previous hierarchical attention + FNO decoder stack with
an entirely convolutional multigrid-attention backbone. The public entrypoint is
still ``HANO2d`` so existing training code can keep importing the same class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Each pair is (pre_smoothing_steps, post_smoothing_steps) for one multigrid level.
DEFAULT_NUM_ITERATION = ((1, 0), (1, 0), (1, 0))


def _resolve_group_count(num_channels, preferred_groups=4):
    """Pick a valid GroupNorm group count for the requested channel size."""
    for num_groups in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class Conv2dAttention(nn.Module):
    """Local attention update driven by a learned key/value convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.key_value_projection = nn.Conv2d(
            in_channels,
            out_channels * 4,
            kernel_size,
            stride,
            padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.output_norm = nn.GroupNorm(_resolve_group_count(out_channels), out_channels, affine=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if isinstance(x, tuple):
            state, key, value = x
            key = key.view(key.shape[0], 2, key.shape[1] // 2, key.shape[2], key.shape[3])
            value = value.view(value.shape[0], 2, value.shape[1] // 2, value.shape[2], value.shape[3])
        else:
            state = x
            key_value = self.key_value_projection(state)
            key_value = key_value.view(
                key_value.shape[0],
                2,
                2,
                self.out_channels,
                key_value.shape[2],
                key_value.shape[3],
            )
            key = key_value[:, 0, ...]
            value = key_value[:, 1, ...]

        attention_scores = torch.einsum("bchw,bgchw->bghw", state, key)
        attention_weights = self.softmax(attention_scores)
        updated_state = torch.einsum("bghw,bgchw->bchw", attention_weights, value)
        updated_state = self.output_norm(updated_state)

        flat_key = key.view(key.shape[0], self.out_channels * 2, key.shape[3], key.shape[4])
        flat_value = value.view(value.shape[0], self.out_channels * 2, value.shape[3], value.shape[4])
        return updated_state, flat_key, flat_value


class RestrictionBlock(nn.Module):
    """Downsample the multigrid state, keys, and values for the next level."""

    def __init__(self, state_downsample, key_downsample, value_downsample):
        super().__init__()
        self.state_downsample = state_downsample
        self.key_downsample = key_downsample
        self.value_downsample = value_downsample

    def forward(self, x):
        state, key, value = x
        return (
            self.state_downsample(state),
            self.key_downsample(key),
            self.value_downsample(value),
        )


class TupleIdentity(nn.Module):
    """Identity helper that makes tuple-valued stages explicit."""

    def forward(self, x):
        return x


class MultigridAttentionBlock(nn.Module):
    """Encoder-decoder style multigrid block used by the updated HANO model."""

    def __init__(
        self,
        num_iterations,
        num_state_channels,
        num_feature_channels,
        padding_mode="zeros",
        bias=True,
    ):
        super().__init__()
        self.num_iterations = [tuple(iteration) for iteration in num_iterations]
        self.num_levels = len(self.num_iterations)
        self.num_state_channels = num_state_channels

        self.pre_smoothers = nn.ModuleList()
        self.post_smoothers = nn.ModuleList()
        self.restrictions = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level, (pre_smooth_steps, post_smooth_steps) in enumerate(self.num_iterations):
            level_channels = (level + 1) * num_feature_channels
            if level == 0 and pre_smooth_steps < 1:
                raise ValueError(
                    "The first multigrid level requires at least one pre-smoothing step "
                    "(set num_iterations[0][0] >= 1 in the config)."
                )

            self.pre_smoothers.append(
                nn.Sequential(
                    *[
                        Conv2dAttention(
                            level_channels,
                            level_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=bias,
                            padding_mode=padding_mode,
                        )
                        for _ in range(pre_smooth_steps)
                    ]
                )
            )

            post_smoothers = [
                Conv2dAttention(
                    level_channels,
                    level_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    padding_mode=padding_mode,
                )
                for _ in range(post_smooth_steps)
            ]
            self.post_smoothers.append(nn.Sequential(*post_smoothers) if post_smoothers else TupleIdentity())

            if level < self.num_levels - 1:
                current_channels = (level + 1) * num_state_channels
                next_channels = (level + 2) * num_state_channels
                self.restrictions.append(
                    RestrictionBlock(
                        state_downsample=nn.Conv2d(
                            current_channels,
                            next_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=bias,
                            padding_mode=padding_mode,
                        ),
                        key_downsample=nn.Conv2d(
                            current_channels * 2,
                            next_channels * 2,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=bias,
                            padding_mode=padding_mode,
                        ),
                        value_downsample=nn.Conv2d(
                            current_channels * 2,
                            next_channels * 2,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=bias,
                            padding_mode=padding_mode,
                        ),
                    )
                )
                self.upsamplers.append(
                    nn.ConvTranspose2d(
                        next_channels,
                        current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=bias,
                    )
                )

    def forward(self, x):
        multigrid_states = [None] * self.num_levels
        current = x

        # Descend through the pyramid while storing skip connections at each level.
        for level in range(self.num_levels):
            current = self.pre_smoothers[level](current)
            multigrid_states[level] = current
            if level < self.num_levels - 1:
                current = self.restrictions[level](current)

        # Reconstruct the finest resolution with transpose-conv upsampling + post smoothing.
        for level in range(self.num_levels - 2, -1, -1):
            state, key, value = multigrid_states[level]
            coarse_state = multigrid_states[level + 1][0]
            refined_state = state + self.upsamplers[level](coarse_state)
            multigrid_states[level] = self.post_smoothers[level]((refined_state, key, value))

        return multigrid_states[0][0]


class HANO2d(nn.Module):
    """Hierarchical Attention Neural Operator with a multigrid-attention backbone."""

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = kwargs.pop("r_dic", None)
        elif "r_dic" in kwargs:
            merged_config = dict(kwargs.pop("r_dic"))
            merged_config.update(config)
            config = merged_config

        if config is None:
            config = kwargs.pop("config", None)

        config = {} if config is None else dict(config)
        if kwargs:
            config.update(kwargs)

        self.boundary_condition = config.get("boundary_condition")
        self.input_channels = config.get("in_dim", 1)
        self.latent_channels = config.get("feature_dim", 24)
        self.output_dim = config.get("output_dim", 1)
        self.num_layers = config.get("num_layers", config.get("num_layer", 1))
        self.num_iterations = self._resolve_num_iterations(config)
        self.normalizer = config.get("y_norm")
        self.use_input_residual = config.get("use_input_residual", False)

        # Preserve explicit padding settings; otherwise choose a sensible default
        # from the boundary condition used by the PDE experiment.
        padding_mode = config.get("padding_mode")
        if padding_mode is None:
            padding_mode = "zeros" if self.boundary_condition == "dirichlet" else "circular"
        self.padding_mode = padding_mode

        self.patch_embedding = nn.Conv2d(
            self.input_channels,
            self.latent_channels,
            kernel_size=3,
            padding=1,
            bias=config.get("bias", False),
            padding_mode=self.padding_mode,
        )
        self.multigrid_blocks = nn.ModuleList(
            [
                MultigridAttentionBlock(
                    num_iterations=self.num_iterations,
                    num_state_channels=self.latent_channels,
                    num_feature_channels=self.latent_channels,
                    padding_mode=self.padding_mode,
                    bias=config.get("bias", False),
                )
                for _ in range(self.num_layers)
            ]
        )

        last_layer = config.get("last_layer", "conv") or "conv"
        if last_layer == "conv":
            self.output_projection = nn.Conv2d(
                self.latent_channels,
                self.output_dim,
                kernel_size=3,
                padding=1,
                bias=config.get("bias", False),
                padding_mode=self.padding_mode,
            )
        elif last_layer == "linear":
            self.output_projection = nn.Conv2d(self.latent_channels, self.output_dim, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer: must be "conv" or "linear"')

        self.activation = self._build_activation(config.get("activation", "gelu"))

        if self.use_input_residual:
            self.input_projection = nn.Conv2d(self.input_channels, self.output_dim, kernel_size=1, bias=False)
        else:
            self.input_projection = None

    @staticmethod
    def _build_activation(name):
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "silu":
            return nn.SiLU()
        raise NameError('invalid activation: must be one of "relu", "gelu", "tanh", or "silu"')

    @staticmethod
    def _resolve_num_iterations(config):
        if "num_iterations" in config:
            return [tuple(iteration) for iteration in config["num_iterations"]]
        if "num_iteration" in config:
            # Keep supporting the legacy singular key until older experiment
            # scripts and checkpoints have fully migrated.
            return [tuple(iteration) for iteration in config["num_iteration"]]

        depths = config.get("depths")
        if depths:
            # Legacy experiment files described each level with `depths`. The new
            # block interprets that count as pre-smoothing steps and defaults the
            # post-smoothing count to zero for backward compatibility.
            return [(int(depth), 0) for depth in depths]

        return [tuple(iteration) for iteration in DEFAULT_NUM_ITERATION]

    def forward(self, x):
        features = self.patch_embedding(x)

        # Each multigrid block mixes local attention updates with coarse-to-fine refinement.
        for block in self.multigrid_blocks:
            features = self.activation(block(features))

        output = self.output_projection(features)

        if self.input_projection is not None:
            output = output + self.input_projection(x)

        # The stored dataset normalizer operates on scalar solution fields, so we
        # only apply it when the model is configured to predict a single channel.
        if self.normalizer is not None and self.output_dim == 1:
            output = self.normalizer.decode(output.squeeze(1)).unsqueeze(1)

        if self.boundary_condition == "dirichlet" and output.shape[-2] > 2 and output.shape[-1] > 2:
            interior = output[:, :, 1:-1, 1:-1].contiguous()
            output = F.pad(interior, (1, 1, 1, 1), "constant", 0)

        return output.permute(0, 2, 3, 1)


# Backward-compatible aliases for earlier names that may still appear in old notebooks.
Conv2dAttn = Conv2dAttention
Restrict = RestrictionBlock
MgConv_DC_3 = MultigridAttentionBlock
HANO = HANO2d
