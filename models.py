"""Backward-compatible imports for model classes."""

from hano.models import DilResNet, FNO2d, HANO, HANO2d, LegacyHANO, LegacyHANO2d
from hano.models.baselines import dCNN
from hano.models.components import (
    DecomposeLayer,
    FeedForward,
    Mlp,
    PatchEmbed,
    PatchMerging,
    ReduceLayer,
    WindowAttention,
    window_partition,
    window_reverse,
)
from hano.models.fno import SpectralConv2d_FNO
from hano.models.hano import (
    Conv2dAttention,
    MgConv_DC_3,
    MultigridAttentionBlock,
    Restrict,
    RestrictionBlock,
)
from hano.models.hano_legacy import Decodermap, HAttention, SpectralConv2d

SpectralDecoder = Decodermap
HTransformer = HAttention

__all__ = [
    "HANO2d",
    "HANO",
    "LegacyHANO2d",
    "LegacyHANO",
    "FNO2d",
    "DilResNet",
    "dCNN",
    "Mlp",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "PatchEmbed",
    "PatchMerging",
    "ReduceLayer",
    "DecomposeLayer",
    "FeedForward",
    "Conv2dAttention",
    "MultigridAttentionBlock",
    "RestrictionBlock",
    "Restrict",
    "MgConv_DC_3",
    "SpectralConv2d",
    "SpectralDecoder",
    "HTransformer",
    "Decodermap",
    "HAttention",
    "SpectralConv2d_FNO",
]
