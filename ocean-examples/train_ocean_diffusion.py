#!/usr/bin/env python
"""
Minimal example: train a diffusion model on ocean data using cBottle's
grid-based UNet (SongUNet with ``mixing_type="spatial"`` + ``Plane`` domain).

The network operates on tensors of shape ``(B, C, T, H*W)`` internally, where
  B = batch, C = channels, T = time frames, H*W = spatial pixels.

This script demonstrates:
  1. A dummy ``OceanDataset`` that returns (channel, time, height, width) data —
     swap this out for your real zarr-backed loader later.
  2. Building a ``SongUNet`` + ``EDMPrecond`` for standard 2-D grids.
  3. A training loop using ``cbottle.loss.EDMLoss``.
  4. **Wandb logging** for loss curves, sigma distributions, and throughput.
  5. **Two conditioning mechanisms** built into cBottle:
     a) Scalar / vector conditioning via ``class_labels`` (label_dim)
     b) Spatial conditioning via ``condition`` channels
  6. **Temporal attention** for multi-frame (video) ocean state prediction.
  7. **Per-frame time-offset encoding** for irregular temporal spacing.

How cBottle handles conditioning
================================

**Scalar conditioning (class_labels / label_dim)**
  The ``SongUNet`` constructor creates a ``Linear(label_dim → noise_channels)``
  layer called ``map_label``.  During the forward pass the class-label vector
  ``(B, label_dim)`` is projected into the same embedding space as the
  diffusion-time embedding and *added* to it:

      emb = map_noise(σ)           # positional / Fourier encoding of σ
      emb = emb + map_label(y)     # add projected label vector

  This combined embedding is then injected into every residual block via
  adaptive group-norm (scale + shift), giving the UNet *global* knowledge of
  the scalar/vector conditioning.

  **Classifier-free guidance** is supported natively: set ``label_dropout > 0``
  and, during training, the label vector is randomly zeroed out with that
  probability.  At inference time you can combine the conditional and
  unconditional predictions with a guidance scale.

**Spatial conditioning (condition channels)**
  ``EDMPrecond.forward`` concatenates the condition tensor *channel-wise* with
  the noise-scaled input before feeding it to the UNet:

      arg = cat([c_in * x, condition], dim=1)   # channel concat
      out = model(arg, ...)

  So the UNet's ``in_channels`` must equal ``out_channels + condition_channels``.
  The condition channels are *not* scaled by ``c_in`` — they are passed as-is,
  which is appropriate for things like observation masks, bathymetry, or
  low-resolution fields that don't participate in the diffusion noise schedule.

**Temporal attention**
  ``SongUNet`` has built-in temporal attention via three parameters:

  - ``temporal_attention_resolutions``:  list of UNet resolutions (e.g. [32, 16])
      at which to add ``TemporalAttention`` blocks in the encoder.  These blocks
      perform self-attention across the T frames at each spatial location,
      with learned *relative* positional embeddings, enabling the network to
      learn temporal correlations between frames.

  - ``decoder_start_with_temporal_attention``:  if True, the bottleneck decoder
      block also includes temporal attention.

  - ``upsample_temporal_attention``:  if True, each upsampling decoder block
      includes temporal attention.

  The temporal attention uses the shape ``(B, C, T, X)`` — frames are the
  sequence dimension, and attention is performed independently per pixel.

**Per-frame time-offset encoding (for irregular temporal spacing)**
  cBottle does not have a built-in "irregular timestep" encoder, but we can
  encode this information using *extra condition channels*.  For each frame
  we provide a scalar time-offset (e.g. hours since reference time), which we
  broadcast to the full spatial grid ``(1, T, H*W)`` and concatenate as an
  additional condition channel.  This way the network "sees" how far apart
  each frame is at every pixel, even when SSH and SST observations arrive at
  different times.

  See ``TEMPORAL_OFFSET_CHANNELS`` below.

Usage (single GPU):
    python train_ocean_diffusion.py

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 train_ocean_diffusion.py
"""

import sys
import os
import copy
import time

import torch
import torch.utils.data
import numpy as np

# ── cBottle imports ──────────────────────────────────────────────────────────
from cbottle.models.networks import SongUNet, EDMPrecond
from cbottle.domain import Plane
from cbottle.loss import EDMLoss
from cbottle.datasets.base import BatchInfo, TimeUnit
from cbottle.datasets.samplers import InfiniteSequentialSampler
import cbottle.distributed

# ── Weights & Biases (optional) ─────────────────────────────────────────────
try:
    import wandb
except ImportError:
    wandb = None

# ── OmegaConf (structured config + CLI overrides) ───────────────────────────
from dataclasses import dataclass, field
from omegaconf import OmegaConf

# ── llc4320 data loader (optional — only needed when use_real_data=True) ────
try:
    sys.path.insert(0, os.path.expanduser(
        "~/shaferlab/tatsu/NYU_SWOT_project/Inpainting_Pytorch_gen/"
        "SWOT-inpainting-DL/src"))
    from claude_data_loaders import llc4320_dataset
except ImportError:
    llc4320_dataset = None


# ════════════════════════════════════════════════════════════════════════════
# 1.  Configuration  (module-level constants — all fed into Config below)
# ════════════════════════════════════════════════════════════════════════════

# Spatial grid (height × width)
H, W = 128, 128

# Number of ocean variable channels (e.g. SST, SSH, U, V, …)
NUM_CHANNELS = 2

# ── Spatial conditioning ─────────────────────────────────────────────────
# Condition channels are *concatenated* to the noisy input along the channel
# axis inside EDMPrecond.  Use this for pixel-level side information such as:
#   • observation masks     (1 channel: 0/1 per pixel)
#   • bathymetry / land-sea mask
#   • low-resolution field to super-resolve
# Set to 0 for unconditional generation.
CONDITION_CHANNELS = 2

# ── Per-frame temporal offset channels ───────────────────────────────────
# Number of extra condition channels that encode *when* each frame was
# observed.  Because SSH and SST observations can arrive at different
# (irregular) times we need to communicate the temporal spacing to the
# network on a per-frame, per-pixel basis.
#
# Approach: for each sample we build a tensor of shape
#   (TEMPORAL_OFFSET_CHANNELS, T, H*W)
# and concatenate it to the spatial condition.  The simplest version uses
# 1 channel = raw time-offset (hours since reference), but you can also use
# a Fourier/sinusoidal expansion for richer representation:
#   • 1 channel  → raw Δt (normalised)
#   • N channels → [sin(ω₁Δt), cos(ω₁Δt), sin(ω₂Δt), cos(ω₂Δt), ...]
# These are *broadcast* across spatial pixels (same time-offset everywhere
# in a frame) so the network learns "this frame is 3 hours old" vs "12 hours".
#
# Set to 0 to disable.
# TEMPORAL_OFFSET_CHANNELS = 4  # sin/cos pair × 2 frequencies
TEMPORAL_OFFSET_CHANNELS = 0  # disabled — all samples have uniform temporal spacing

# Total condition channels seen by the UNet (spatial + temporal offsets)
TOTAL_CONDITION_CHANNELS = CONDITION_CHANNELS + TEMPORAL_OFFSET_CHANNELS

# ── Scalar / vector conditioning ─────────────────────────────────────────
# label_dim is the size of the per-sample label vector that gets projected
# into the time-embedding space.  Use this to encode *global* metadata like:
#   • dataset source ID  (one-hot: ERA5 vs. ICON vs. obs)
#   • season / month     (one-hot or cyclical encoding)
#   • lead time          (scalar)
#   • ensemble member ID
# Set to 0 for unconditional generation.
LABEL_DIM = 8

# Dropout probability for class labels during training.  Setting this > 0
# enables classifier-free guidance at inference time: the model learns to
# generate both conditionally (labels present) and unconditionally (labels
# zeroed), and you interpolate between the two predictions at sampling.
LABEL_DROPOUT = 0.1

# ── Temporal / video settings ────────────────────────────────────────────
# Time frames per sample.  Set >1 to model the *evolution* of ocean fields
# (e.g. 5 consecutive SSH/SST snapshots).
TIME_LENGTH = 5

# ── Temporal attention ───────────────────────────────────────────────────
# SongUNet has built-in TemporalAttention that performs self-attention across
# frames at each spatial location.  It uses *learned relative positional
# embeddings* so the network can distinguish frame ordering.
#
# temporal_attention_resolutions: list of UNet internal resolutions at which
#   to insert TemporalAttention in the encoder.  With channel_mult=[1,2,2,2]
#   and img_resolution=64, the encoder resolutions are 64, 32, 16, 8.
#   Adding attention at [16, 8] means the two coarsest scales get temporal
#   attention — a good cost / quality trade-off.
#
# decoder_start_with_temporal_attention: if True, the decoder bottleneck
#   block also includes temporal attention.
#
# upsample_temporal_attention: if True, every upsampling block in the
#   decoder includes temporal attention (more expensive).
TEMPORAL_ATTN_RESOLUTIONS = [16, 8]
DECODER_START_TEMPORAL_ATTN = False
UPSAMPLE_TEMPORAL_ATTN = False

# Training hyper-parameters
BATCH_SIZE = 80
LR = 2e-4
NUM_STEPS = 2000000
LOG_EVERY = 50
EMA_DECAY = 0.999

# UNet capacity
MODEL_CHANNELS = 64  # keep small for a demo; increase for real training

# Wandb
WANDB_PROJECT = "ocean-diffusion"
WANDB_ENABLED = True  # set False to disable

# ── Calendar conditioning ────────────────────────────────────────────────
# cBottle's SongUNet has a built-in CalendarEmbedding that converts
# second_of_day and day_of_year into learned sinusoidal features and
# concatenates them as extra input channels.  Set > 0 to enable.
# Typical values: 4–16.  Each of second_of_day and day_of_year produces
# calendar_embed_channels/2 channels, so the total extra input channels
# added to the UNet = calendar_embed_channels.
CALENDAR_EMBED_CHANNELS = 8

# ── Real data (llc4320) ──────────────────────────────────────────────────
# Set USE_REAL_DATA=True to swap the dummy OceanDataset for the llc4320
# adapter.  The fields below mirror the constructor args of llc4320_dataset
# and must be configured for your filesystem layout.
USE_REAL_DATA = True
DATA_DIR = "/scratch/tm3076/greene_vast/pytorch_learning_tiles"                         # root of llc4320 zarr / npy tiles
PATCH_COORDS_PATH = f"{DATA_DIR}/FULL_PACIFIC_coords.npy"                # .npy with (N_patches, 3) coords
MID_TIMESTEP = 100                    # central timestep index (ignored when use_concat_time=True)
INFIELDS  = ["FULL_PACIFIC_zarr_llc4320_SSH_minus_300km_filtered_fullt", "FULL_PACIFIC_zarr_llc4320_SST_tiles_4km_unfiltered_fullt"]  # input variable zarr directory names
OUTFIELDS = ["FULL_PACIFIC_zarr_llc4320_SSH_minus_300km_filtered_fullt", "FULL_PACIFIC_zarr_llc4320_SST_tiles_4km_unfiltered_fullt"]  # output (target) variable names
IN_MASK_LIST  = ["None", "None"]                 # mask key per input field
OUT_MASK_LIST = ["None", "None"]                 # mask key per output field
IN_TRANSFORM_LIST  = ["std_ssh_norm", "std_seasonal_mean_sst_norm"]  # transform key per input field
OUT_TRANSFORM_LIST = ["std_ssh_norm", "std_seasonal_mean_sst_norm"]  # transform key per output field
LLC_STANDARDS = None                  # dict of normalisation constants, or None for defaults
LLC_N = 128                           # spatial tile size (pixels)
LLC_L_X = 512e3                       # tile extent in metres (x)
LLC_L_Y = 512e3                       # tile extent in metres (y)
LLC_DTIME = 12                        # temporal stride (hours)

# ── Temporal coverage via ConcatDataset ──────────────────────────────────
# Like train_diff_CLEAN.py: create one llc4320_dataset per mid_timestep and
# concatenate them so the model sees data from the full temporal range.
# Each sub-dataset also has randomize_time=True for ±12h local jitter.
# Time range is in *hourly* timestep indices (0..9029 for llc4320).
# With dtime=12 and N_t=5, the half-window is 12*2=24, so valid mid_t is
# [24, 9005].  We sample every TIME_RANGE_STEP hours.
USE_CONCAT_TIME = True
TIME_RANGE_START = 48       # first mid_timestep
TIME_RANGE_END = 9000       # last mid_timestep (exclusive)
TIME_RANGE_STEP = 80       # step between mid_timesteps (~5 days)


# ════════════════════════════════════════════════════════════════════════════
# 1b. Structured config  (populated from the constants above)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """All tuneable parameters in one place.

    Defaults are drawn from the module-level constants so the script behaves
    identically with or without CLI overrides.  Override any field from the
    command line:

        python train_ocean_diffusion.py H=128 W=128 lr=1e-4

    Or load a YAML file and merge:

        cfg = OmegaConf.merge(cfg, OmegaConf.load("my_experiment.yaml"))
    """
    # ── Grid ──
    H: int = H
    W: int = W

    # ── Channels ──
    num_channels: int = NUM_CHANNELS
    condition_channels: int = CONDITION_CHANNELS
    temporal_offset_channels: int = TEMPORAL_OFFSET_CHANNELS
    #total_condition_channels: int = CONDITION_CHANNELS + TEMPORAL_OFFSET_CHANNELS

    # ── Label / scalar conditioning ──
    label_dim: int = LABEL_DIM
    label_dropout: float = LABEL_DROPOUT

    # ── Temporal ──
    time_length: int = TIME_LENGTH
    temporal_attn_resolutions: list = field(default_factory=lambda: list(TEMPORAL_ATTN_RESOLUTIONS))
    decoder_start_temporal_attn: bool = DECODER_START_TEMPORAL_ATTN
    upsample_temporal_attn: bool = UPSAMPLE_TEMPORAL_ATTN

    # ── Training ──
    batch_size: int = BATCH_SIZE
    lr: float = LR
    num_steps: int = NUM_STEPS
    log_every: int = LOG_EVERY
    ema_decay: float = EMA_DECAY

    # ── Model ──
    model_channels: int = MODEL_CHANNELS
    calendar_embed_channels: int = CALENDAR_EMBED_CHANNELS

    # ── Wandb ──
    wandb_project: str = WANDB_PROJECT
    wandb_enabled: bool = WANDB_ENABLED

    # ── Real data (llc4320) ──
    use_real_data: bool = USE_REAL_DATA
    data_dir: str = DATA_DIR
    patch_coords_path: str = PATCH_COORDS_PATH
    mid_timestep: int = MID_TIMESTEP
    infields: list = field(default_factory=lambda: list(INFIELDS))
    outfields: list = field(default_factory=lambda: list(OUTFIELDS))
    in_mask_list: list = field(default_factory=lambda: list(IN_MASK_LIST))
    out_mask_list: list = field(default_factory=lambda: list(OUT_MASK_LIST))
    in_transform_list: list = field(default_factory=lambda: list(IN_TRANSFORM_LIST))
    out_transform_list: list = field(default_factory=lambda: list(OUT_TRANSFORM_LIST))
    llc_standards: dict = field(default_factory=lambda: LLC_STANDARDS or {})
    llc_n: int = LLC_N
    llc_l_x: float = LLC_L_X
    llc_l_y: float = LLC_L_Y
    llc_dtime: int = LLC_DTIME

    # ── Temporal coverage via ConcatDataset ──
    use_concat_time: bool = USE_CONCAT_TIME
    time_range_start: int = TIME_RANGE_START
    time_range_end: int = TIME_RANGE_END
    time_range_step: int = TIME_RANGE_STEP

# Build the config: structured defaults + CLI overrides
# e.g.  python train_ocean_diffusion.py H=128 lr=1e-4 wandb_enabled=false
_base_cfg = OmegaConf.structured(Config)
_cli_cfg = OmegaConf.from_cli()
cfg: Config = OmegaConf.merge(_base_cfg, _cli_cfg)  # type: ignore[assignment]


# ════════════════════════════════════════════════════════════════════════════
# 2.  Helpers
# ════════════════════════════════════════════════════════════════════════════

def sinusoidal_time_encoding(offsets: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Encode scalar time offsets into sinusoidal features.

    Args:
        offsets:    (T,) — time offset per frame (e.g. hours since reference)
        n_channels: number of output channels (must be even)

    Returns:
        (n_channels, T) — sin/cos encoding broadcast-ready for spatial dims
    """
    assert n_channels % 2 == 0, "n_channels must be even for sin/cos pairs"
    n_freq = n_channels // 2
    # Log-spaced frequencies: longest period ≈ 720 h (30 days), shortest ≈ 1 h
    freqs = torch.logspace(-1, 2, n_freq)                       # (n_freq,)
    # (n_freq, T) = outer product
    angles = 2.0 * torch.pi * freqs[:, None] * offsets[None, :] / 720.0
    return torch.cat([angles.sin(), angles.cos()], dim=0)       # (n_channels, T)


# ════════════════════════════════════════════════════════════════════════════
# 3.  Dummy ocean dataset  (replace with your zarr / netcdf loader)
# ════════════════════════════════════════════════════════════════════════════

class OceanDataset(torch.utils.data.Dataset):
    """Dummy dataset that returns random tensors shaped like ocean fields.

    Each sample is a dict with:
        target:          (C, T, H*W)                  — the field(s) to denoise
        condition:       (C_cond + C_time, T, H*W)    — spatial + temporal cond
        class_labels:    (label_dim,)                  — scalar / vector conditioning
        second_of_day:   (T,)
        day_of_year:     (T,)

    NOTE: cBottle's SongUNet (with ``mixing_type="spatial"``) expects the
    spatial dimensions to be *flattened* into a single axis, i.e.
    ``(B, C, T, H*W)``.  The ``SpatialFactory`` internally reshapes to
    ``(B*T, C, H, W)`` for standard 2-D convolutions and back.
    """

    def __init__(self, cfg: Config, num_samples: int = 1000, split: str = "train"):
        self.cfg = cfg
        self.num_samples = num_samples
        self.split = split
        self.batch_info = BatchInfo(
            channels=[f"ocean_var_{i}" for i in range(cfg.num_channels)],
            time_step=1,
            time_unit=TimeUnit.HOUR,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        c = self.cfg
        # ── Replace this block with real data loading ──
        # For example, read a zarr array with shape (C, T, H, W), then
        # flatten the last two dims to (C, T, H*W).
        target = torch.randn(c.num_channels, c.time_length, c.H * c.W)
        # ── End placeholder ──

        sample = {
            "target": target,
            # Temporal encodings (used by calendar embedding if enabled)
            "second_of_day": torch.zeros(c.time_length),
            "day_of_year": torch.zeros(c.time_length),
        }

        # ── Scalar / vector labels ──
        if c.label_dim > 0:
            labels = torch.zeros(c.label_dim)
            labels[idx % c.label_dim] = 1.0  # dummy one-hot
            sample["class_labels"] = labels

        # ── Build spatial + temporal-offset condition ─────────────────────
        cond_parts = []

        # a) Pixel-level spatial condition (obs mask, bathymetry, …)
        if c.condition_channels > 0:
            cond_parts.append(
                torch.randn(c.condition_channels, c.time_length, c.H * c.W)
            )

        # b) Per-frame temporal offset encoding
        #    In real usage: offsets_hours = actual observation times in hours
        #    relative to some reference (e.g. first frame).
        #    SSH might be at [0, 12, 24, 48, 72] hours
        #    SST might be at [0,  6, 18, 36, 60] hours
        #    You'd encode each channel's offset schedule and concatenate.
        if c.temporal_offset_channels > 0:
            # Dummy: irregularly spaced frames (hours since t=0)
            offsets_hours = torch.sort(torch.rand(c.time_length) * 72.0).values
            time_enc = sinusoidal_time_encoding(offsets_hours, c.temporal_offset_channels)
            # Broadcast from (C_time, T) → (C_time, T, H*W)
            cond_parts.append(
                time_enc.unsqueeze(-1).expand(-1, -1, c.H * c.W)
            )

        if cond_parts:
            sample["condition"] = torch.cat(cond_parts, dim=0)

        return sample


# ────────────────────────────────────────────────────────────────────────────
# 3b.  llc4320 adapter — wraps llc4320_dataset to produce cBottle-format dicts
# ────────────────────────────────────────────────────────────────────────────

class LLC4320Adapter(torch.utils.data.Dataset):
    """Wrap ``llc4320_dataset`` so it returns the dict format cBottle expects.

    ``llc4320_dataset.__getitem__`` returns a tuple::

        (invar, outvar)                          # return_masks=False
        (invar, outvar, inmask, outmask)         # return_masks=True

    where each tensor has shape ``(N_t, C_fields, N, N)`` — i.e.
    **(time, channels, height, width)**.

    cBottle needs a dict with::

        target:       (C, T, H*W)
        condition:    (C_cond, T, H*W)     # optional
        class_labels: (label_dim,)         # optional
        second_of_day: (T,)
        day_of_year:   (T,)

    The adapter:
      • Permutes ``outvar`` from (T, C, H, W) → (C, T, H*W).
      • Builds ``condition`` by concatenating ``invar * inmask`` (the masked
        observations) and the observation masks themselves, all flattened to
        (C_cond, T, H*W).  If temporal-offset channels are configured the
        sinusoidal time encoding is appended as well.
      • Provides dummy ``class_labels`` / calendar tensors.
    """

    def __init__(self, cfg: "Config"):
        if llc4320_dataset is None:
            raise ImportError(
                "llc4320_dataset not found.  Make sure claude_data_loaders.py "
                "is on sys.path (see the import block at the top of this script)."
            )

        patch_coords = np.load(cfg.patch_coords_path)[:-7,:]
        standards = dict(cfg.llc_standards) if cfg.llc_standards else None

        common_kwargs = dict(
            data_dir=cfg.data_dir,
            N_t=cfg.time_length,
            patch_coords=patch_coords,
            infields=list(cfg.infields),
            outfields=list(cfg.outfields),
            in_mask_list=list(cfg.in_mask_list),
            out_mask_list=list(cfg.out_mask_list),
            in_transform_list=list(cfg.in_transform_list),
            out_transform_list=list(cfg.out_transform_list),
            standards=standards,
            N=cfg.llc_n,
            L_x=cfg.llc_l_x,
            L_y=cfg.llc_l_y,
            squeeze=False,
            return_masks=True,
            dtime=cfg.llc_dtime,
            randomize_time=True,
            time_jitter_hours=cfg.llc_dtime,  # ±dtime hours jitter
        )

        if cfg.use_concat_time:
            # Create one sub-dataset per mid_timestep, concatenate for full
            # temporal coverage — same approach as train_diff_CLEAN.py.
            time_range = range(cfg.time_range_start, cfg.time_range_end, cfg.time_range_step)
            sub_datasets = [
                llc4320_dataset(mid_timestep=mid_t, **common_kwargs)
                for mid_t in time_range
            ]
            self._ds = torch.utils.data.ConcatDataset(sub_datasets)
            # Grab one sub-dataset for metadata
            self._representative_ds = sub_datasets[0]
        else:
            self._ds = llc4320_dataset(mid_timestep=cfg.mid_timestep, **common_kwargs)
            self._representative_ds = self._ds

        self.cfg = cfg

        # The number of input-field channels (each field × mask = 2 per field)
        n_in_fields = len(cfg.infields)
        # condition = masked_input (n_in_fields) + masks (n_in_fields)
        self._cond_channels = 2 * n_in_fields

        # Expose batch_info so the training loop can query channel names
        out_names = list(cfg.outfields)
        self.batch_info = BatchInfo(
            channels=out_names,
            time_step=cfg.llc_dtime,
            time_unit=TimeUnit.HOUR,
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        c = self.cfg
        # llc4320_dataset with return_masks=True returns:
        #   (invar, outvar, inmask, outmask, time_coords)
        # where time_coords is a dict with "second_of_day" and "day_of_year"
        #   each shaped (N_t, C_fields, N, N)
        raw = self._ds[idx]
        # The last element is always the time_coords dict
        time_coords = raw[-1]
        invar, outvar, inmask, outmask = raw[0], raw[1], raw[2], raw[3]

        # ── target: outvar  (T, C, H, W) → (C, T, H*W) ──
        T, C_out, Hh, Ww = outvar.shape
        target = outvar.permute(1, 0, 2, 3).reshape(C_out, T, Hh * Ww)  # (C, T, H*W)

        # ── condition (only if we have real masks / observations) ────────
        # When all masks are "None" the invar == outvar (redundant) and
        # inmask is all-ones, so we skip building a spatial condition — the
        # model trains unconditionally on outvar alone.
        cond_parts = []

        all_masks_none = all(m.lower() == "none" for m in c.in_mask_list)
        if not all_masks_none:
            # invar has mask already applied; pass it + raw mask as condition
            C_in = invar.shape[1]
            obs = invar.permute(1, 0, 2, 3).reshape(C_in, T, Hh * Ww)
            msk = inmask.permute(1, 0, 2, 3).reshape(C_in, T, Hh * Ww)
            cond_parts.extend([obs, msk])

        # ── temporal-offset encoding (if configured) ─────────────────────
        if c.temporal_offset_channels > 0:
            offsets_hours = torch.arange(T, dtype=torch.float32) * c.llc_dtime
            time_enc = sinusoidal_time_encoding(offsets_hours, c.temporal_offset_channels)
            cond_parts.append(
                time_enc.unsqueeze(-1).expand(-1, T, Hh * Ww)
            )

        # ── Calendar conditioning ──
        # Convert UTC second_of_day to local solar time using the tile's
        # central longitude, matching what CalendarEmbedding would compute:
        #   local_time = (second_of_day + lon * 86400 / 360) % 86400
        # We do this here so the model's CalendarEmbedding (with lon=0)
        # receives already-local times and the per-tile geography is encoded.
        second_of_day = time_coords["second_of_day"]        # (T,)
        day_of_year   = time_coords["day_of_year"]          # (T,)
        central_lon   = time_coords["central_lon"]           # scalar
        local_second  = (second_of_day + central_lon * 86400.0 / 360.0) % 86400.0

        sample: dict = {
            "target": target,
            "second_of_day": local_second,   # (T,) local solar time
            "day_of_year": day_of_year,      # (T,) real calendar
        }

        if cond_parts:
            sample["condition"] = torch.cat(cond_parts, dim=0)

        if c.label_dim > 0:
            sample["class_labels"] = torch.zeros(c.label_dim)

        return sample


def build_dataset(cfg: "Config", split: str = "train") -> torch.utils.data.Dataset:
    """Factory: return the right dataset based on ``cfg.use_real_data``."""
    if cfg.use_real_data:
        return LLC4320Adapter(cfg)
    else:
        return OceanDataset(cfg=cfg, num_samples=2000, split=split)


# ════════════════════════════════════════════════════════════════════════════
# 4.  Build model
# ════════════════════════════════════════════════════════════════════════════

def build_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    """Construct SongUNet + EDMPrecond for a standard 2-D (H x W) grid.

    The UNet's ``in_channels`` is set to ``NUM_CHANNELS + TOTAL_CONDITION_CHANNELS``
    because EDMPrecond concatenates the condition along the channel axis.
    ``label_dim`` controls the size of the global label-vector pathway.

    SongUNet knobs available for customisation
    -------------------------------------------
    Architecture size:
        model_channels   — base channel width (64 → small demo, 128–256 → production)
        channel_mult     — per-resolution multipliers, e.g. [1,2,2,2] → 4 levels
        num_blocks       — residual blocks per resolution (2–4 typical)
        channels_per_head — attention head width (-1 = one head = full channel dim)
        channel_mult_emb — multiplier for the embedding vector dim (default 4)

    Spatial attention:
        attn_resolutions — resolutions where spatial self-attention is used

    Temporal attention (for TIME_LENGTH > 1):
        temporal_attention_resolutions — encoder resolutions with TemporalAttention
        decoder_start_with_temporal_attention — temporal attn in decoder bottleneck
        upsample_temporal_attention — temporal attn in each decoder upsample block

    Encoder / decoder style:
        encoder_type     — 'standard' (DDPM++), 'skip', or 'residual' (NCSN++)
        decoder_type     — 'standard' or 'skip'
        embedding_type   — 'positional' (DDPM++), 'fourier' (NCSN++), 'zero'

    Regularisation:
        dropout          — intermediate activation dropout
        label_dropout    — classifier-free guidance dropout on class_labels

    Position encoding:
        add_spatial_embedding — learnable spatial position embedding
    """
    domain = Plane(nx=cfg.W, ny=cfg.H)

    # For CalendarEmbedding on a Plane domain we need an explicit lon tensor.
    # Use a placeholder (0°) here — the actual per-tile central longitude is
    # irrelevant at model-construction time because CalendarEmbedding only
    # uses lon to compute local solar time, and the lon buffer is fixed.
    # We set it to 0° so the embedding learns UTC-relative time; the real
    # per-tile longitude will shift second_of_day to local time in the
    # dataset adapter instead (TODO for future work).
    # Shape must be (H*W,) — one value per spatial pixel.
    calendar_lon = torch.zeros(cfg.H * cfg.W) if cfg.calendar_embed_channels > 0 else None

    architecture = SongUNet(
        domain=domain,
        # in_channels = target channels + all condition channels (channel-concat)
        in_channels=cfg.num_channels + cfg.condition_channels + cfg.temporal_offset_channels,
        out_channels=cfg.num_channels,
        label_dim=cfg.label_dim,
        label_dropout=cfg.label_dropout,
        model_channels=cfg.model_channels,
        channel_mult=[1, 2, 2, 2],
        num_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        time_length=cfg.time_length,
        # --- key setting for standard grids ---
        mixing_type="spatial",
        # Calendar embedding: converts second_of_day + day_of_year into
        # learned sinusoidal channels that are concatenated to the input.
        # Set > 0 to condition on time-of-day and season.
        calendar_embed_channels=cfg.calendar_embed_channels,
        calendar_lon=calendar_lon,
        # --- temporal attention ---
        temporal_attention_resolutions=cfg.temporal_attn_resolutions if cfg.time_length > 1 else None,
        decoder_start_with_temporal_attention=cfg.decoder_start_temporal_attn and cfg.time_length > 1,
        upsample_temporal_attention=cfg.upsample_temporal_attn and cfg.time_length > 1,
    )

    net = EDMPrecond(
        model=architecture,
        domain=domain,
        img_channels=cfg.num_channels,
        time_length=cfg.time_length,
        label_dim=cfg.label_dim,
        condition_channels=cfg.condition_channels + cfg.temporal_offset_channels,
    )

    return net.to(device)


# ════════════════════════════════════════════════════════════════════════════
# 5.  Training loop
# ════════════════════════════════════════════════════════════════════════════

def train():
    # ── Distributed init (works for single-GPU too) ──
    cbottle.distributed.init()
    rank = cbottle.distributed.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_output_dir = os.path.join(os.path.dirname(__file__), "outputs", timestamp)
        checkpoints_dir = os.path.join(base_output_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f"[INFO] Outputs will be written to: {base_output_dir}")
    else:
        base_output_dir = None
        checkpoints_dir = None
    # Broadcast output dirs to all ranks
    #base_output_dir = cbottle.distributed.broadcast_object(base_output_dir)
    #checkpoints_dir = cbottle.distributed.broadcast_object(checkpoints_dir)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())

    # ── Wandb init (rank 0 only) ──
    use_wandb = cfg.wandb_enabled and wandb is not None and rank == 0
    if use_wandb:
        try:
            wandb.init(
                project=cfg.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=["ocean", "diffusion", "cbottle"],
            )
        except wandb.errors.CommError as e:
            print(f"[WARN] wandb init failed ({e}), continuing without wandb")
            use_wandb = False

    if rank == 0:
        print(f"Device: {device}")
        print(f"Grid: {cfg.H}x{cfg.W}, Channels: {cfg.num_channels}, Time: {cfg.time_length}")
        print(f"Condition channels: {cfg.condition_channels} (+ {cfg.temporal_offset_channels} temporal = {cfg.condition_channels + cfg.temporal_offset_channels} total)")
        print(f"Label dim: {cfg.label_dim}, Label dropout: {cfg.label_dropout}")
        print(f"Model channels: {cfg.model_channels}, Batch size: {cfg.batch_size}")
        print(f"Calendar embed channels: {cfg.calendar_embed_channels}")
        print(f"Wandb: {'enabled' if use_wandb else 'disabled'}")

    # ── Dataset & DataLoader ──
    train_dataset = build_dataset(cfg, split="train")

    if rank == 0:
        if cfg.use_concat_time:
            n_timesteps = len(range(cfg.time_range_start, cfg.time_range_end, cfg.time_range_step))
            print(f"ConcatDataset: {n_timesteps} timesteps × {len(np.load(cfg.patch_coords_path))} patches = {len(train_dataset)} samples")
        else:
            print(f"Dataset: {len(train_dataset)} samples")

    # When using llc4320 real data the condition channel count is determined
    # by the dataset (2 × n_input_fields + temporal_offset_channels) and the
    # output channel count equals the number of output fields.  Override the
    # config values so the model is built with matching dimensions.
    if cfg.use_real_data and isinstance(train_dataset, LLC4320Adapter):
        n_out = len(cfg.outfields)
        # When all masks are "None", no spatial condition is built from invar,
        # so condition_channels from data = 0.  Otherwise 2 × n_input_fields.
        all_masks_none = all(m.lower() == "none" for m in cfg.in_mask_list)
        data_cond_ch = 0 if all_masks_none else train_dataset._cond_channels
        if n_out != cfg.num_channels:
            if rank == 0:
                print(f"[auto] Overriding num_channels: {cfg.num_channels} → {n_out} (from outfields)")
            cfg.num_channels = n_out  # type: ignore[attr-defined]
        if data_cond_ch != cfg.condition_channels:
            if rank == 0:
                print(f"[auto] Overriding condition_channels: {cfg.condition_channels} → {data_cond_ch} (from infields)")
            cfg.condition_channels = data_cond_ch  # type: ignore[attr-defined]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=InfiniteSequentialSampler(train_dataset),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    train_iter = iter(train_loader)

    # ── Model ──
    net = build_model(cfg, device)
    ema_net = copy.deepcopy(net).eval().requires_grad_(False)

    if rank == 0:
        n_params = sum(p.numel() for p in (
            net._orig_mod.parameters() if hasattr(net, "_orig_mod") else net.parameters()
        ) if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")
        if use_wandb:
            wandb.config.update({"trainable_params": n_params})

    # ── Optimizer & Loss ──
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = EDMLoss(distribution="log_uniform")

    # ── Mixed precision (bf16 on L40S / A100; fp16 fallback) ──
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    if rank == 0:
        print(f"[INFO] AMP: {use_amp}, dtype: {amp_dtype}")

    # ── Training ──
    net.train()
    t0 = time.time()

    for step in range(1, cfg.num_steps + 1):
        batch = next(train_iter)

        # Stage to device
        target = batch["target"].to(device)          # (B, C, T, H*W)

        # ── Scalar conditioning ──
        class_labels = batch.get("class_labels")
        if class_labels is not None and class_labels.numel() > 0:
            class_labels = class_labels.to(device)   # (B, label_dim)
        else:
            class_labels = None

        # ── Spatial conditioning ──
        condition = batch.get("condition")
        if condition is not None:
            condition = condition.to(device)          # (B, C_cond, T, H*W)

        # ── Calendar conditioning ──
        second_of_day = batch.get("second_of_day")
        day_of_year = batch.get("day_of_year")
        if second_of_day is not None:
            second_of_day = second_of_day.to(device)  # (B, T)
        if day_of_year is not None:
            day_of_year = day_of_year.to(device)      # (B, T)

        # EDMLoss expects a callable  net(images, sigma) -> Output
        # We curry the extra args (class_labels, condition) following the same
        # pattern used in cbottle's train_coarse.py  (_curry_net).
        def eval_net(images, sigma):
            return net(
                images, sigma,
                class_labels=class_labels,
                condition=condition,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
            )

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss_output = loss_fn(eval_net, images=target)
            loss_val = loss_output.total.mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss_val).backward()
        scaler.step(optimizer)
        scaler.update()

        # EMA update
        with torch.no_grad():
            for p_ema, p in zip(ema_net.parameters(), net.parameters()):
                p_ema.lerp_(p, 1.0 - cfg.ema_decay)

        # ── Logging ──
        if rank == 0 and step % cfg.log_every == 0:
            elapsed = time.time() - t0
            imgs_per_sec = (step * cfg.batch_size) / elapsed
            loss_scalar = loss_val.item()
            sigma_mean = loss_output.sigma.mean().item()

            print(
                f"Step {step:5d}/{cfg.num_steps} | "
                f"loss: {loss_scalar:.4f} | "
                f"sigma_mean: {sigma_mean:.2f} | "
                f"{imgs_per_sec:.1f} img/s"
            )

            if use_wandb:
                log_dict = {
                    "train/loss": loss_scalar,
                    "train/denoising_loss": loss_output.denoising.mean().item(),
                    "train/sigma_mean": sigma_mean,
                    "train/sigma_min": loss_output.sigma.min().item(),
                    "train/sigma_max": loss_output.sigma.max().item(),
                    "perf/imgs_per_sec": imgs_per_sec,
                    "step": step,
                }
                if loss_output.classification is not None:
                    log_dict["train/classification_loss"] = (
                        loss_output.classification.mean().item()
                    )
                wandb.log(log_dict, step=step)

    # ── Save checkpoint ──
    if rank == 0:
        ckpt_path = os.path.join(checkpoints_dir, "ocean_diffusion_checkpoint.pt")
        torch.save(
            {
                "step": cfg.num_steps,
                "model_state_dict": net.state_dict(),
                "ema_state_dict": ema_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        total_time = time.time() - t0
        print(f"\nCheckpoint saved to {ckpt_path}")
        print(f"Total training time: {total_time:.1f}s")

        if use_wandb:
            wandb.log({"total_training_time_s": total_time}, step=cfg.num_steps)
            wandb.finish()


# ════════════════════════════════════════════════════════════════════════════
# 6.  Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
