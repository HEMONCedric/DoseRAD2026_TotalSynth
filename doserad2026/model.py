"""Final energy-conditioned residual 3-D U-Net used for proton BEV dose."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SharedEnergyFiLM(nn.Module):
    """Shared energy encoder with zero-initialized heads for residual FiLM."""

    def __init__(self, channels_by_stage: dict[str, int], hidden_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_channels, 2 * channels) for name, channels in channels_by_stage.items()}
        )
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def embed(self, energy: torch.Tensor) -> torch.Tensor:
        return self.encoder(energy.float())

    def forward(self, x: torch.Tensor, embedding: torch.Tensor, stage: str) -> torch.Tensor:
        gamma, beta = self.heads[stage](embedding).to(dtype=x.dtype).chunk(2, dim=1)
        return x * (1.0 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        film: SharedEnergyFiLM | None = None,
        film_stage: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (film is None) != (film_stage is None):
            raise ValueError("film and film_stage must either both be set or both be None")
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.film = film
        self.film_stage = film_stage
        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv3d(in_channels, out_channels, 1, stride, bias=False)
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.norm1(self.conv1(x))
        if self.film is not None:
            x = self.film(x, embedding, self.film_stage)
        x = self.dropout(F.silu(x, inplace=True))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class DecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        film: SharedEnergyFiLM | None = None,
        film_stage: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.block = ResidualBlock3D(
            out_channels + skip_channels,
            out_channels,
            film=film,
            film_stage=film_stage,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(torch.cat((self.reduce(x), skip), dim=1), embedding)


class ProtonFiLMLiteResUNet3D(nn.Module):
    """Six-level, base-6 ProtonFiLM-Lite network.

    Deep supervision is a training-only option. Deployed checkpoints should be
    exported without ``aux_dec2``/``aux_dec3`` and loaded with the default
    ``deep_supervision=False``.
    """

    def __init__(
        self,
        base_channels: int = 6,
        deep_dropout: float = 0.05,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= deep_dropout < 1.0:
            raise ValueError("deep_dropout must be in [0, 1)")
        c = int(base_channels)
        self.energy_film = SharedEnergyFiLM(
            {
                "enc3": 4 * c,
                "enc4": 8 * c,
                "enc5": 16 * c,
                "bottleneck": 32 * c,
                "dec5": 16 * c,
                "dec4": 8 * c,
            }
        )
        self.stem = ResidualBlock3D(1, c)
        self.enc2 = ResidualBlock3D(c, 2 * c, stride=2)
        self.enc3 = ResidualBlock3D(
            2 * c, 4 * c, stride=2, film=self.energy_film, film_stage="enc3"
        )
        self.enc4 = ResidualBlock3D(
            4 * c, 8 * c, stride=2, film=self.energy_film, film_stage="enc4"
        )
        self.enc5 = ResidualBlock3D(
            8 * c, 16 * c, stride=2, film=self.energy_film, film_stage="enc5"
        )
        self.bottleneck = ResidualBlock3D(
            16 * c,
            32 * c,
            stride=2,
            film=self.energy_film,
            film_stage="bottleneck",
            dropout=deep_dropout,
        )
        self.dec5 = DecoderBlock3D(
            32 * c,
            16 * c,
            16 * c,
            film=self.energy_film,
            film_stage="dec5",
            dropout=deep_dropout,
        )
        self.dec4 = DecoderBlock3D(
            16 * c, 8 * c, 8 * c, film=self.energy_film, film_stage="dec4"
        )
        self.dec3 = DecoderBlock3D(8 * c, 4 * c, 4 * c)
        self.dec2 = DecoderBlock3D(4 * c, 2 * c, 2 * c)
        self.dec1 = DecoderBlock3D(2 * c, c, c)
        self.head = nn.Conv3d(c, 1, 1)
        self.deep_supervision = bool(deep_supervision)
        self.aux_dec2 = nn.Conv3d(2 * c, 1, 1) if self.deep_supervision else None
        self.aux_dec3 = nn.Conv3d(4 * c, 1, 1) if self.deep_supervision else None

        nn.init.normal_(self.head.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.head.bias, -8.0)
        for auxiliary_head in (self.aux_dec2, self.aux_dec3):
            if auxiliary_head is not None:
                nn.init.normal_(auxiliary_head.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(auxiliary_head.bias, -8.0)

    @staticmethod
    def _validate_energy(energy: torch.Tensor, batch_size: int) -> torch.Tensor:
        energy = energy.reshape(batch_size, -1)
        if energy.shape[1] != 1:
            raise ValueError(f"Expected energy shaped (B, 1), got {tuple(energy.shape)}")
        if not torch.isfinite(energy).all():
            raise ValueError("Energy contains non-finite values")
        return energy

    def forward(self, x: torch.Tensor, energy: torch.Tensor):
        energy = self._validate_energy(energy, x.shape[0])
        embedding = self.energy_film.embed(energy)
        x1 = self.stem(x, embedding)
        x2 = self.enc2(x1, embedding)
        x3 = self.enc3(x2, embedding)
        x4 = self.enc4(x3, embedding)
        x5 = self.enc5(x4, embedding)
        x6 = self.bottleneck(x5, embedding)
        x = self.dec5(x6, x5, embedding)
        x = self.dec4(x, x4, embedding)
        x_dec3 = self.dec3(x, x3, embedding)
        x_dec2 = self.dec2(x_dec3, x2, embedding)
        x = self.dec1(x_dec2, x1, embedding)
        prediction = F.softplus(self.head(x))
        if not self.deep_supervision:
            return prediction
        return (
            prediction,
            F.softplus(self.aux_dec2(x_dec2)),
            F.softplus(self.aux_dec3(x_dec3)),
        )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
