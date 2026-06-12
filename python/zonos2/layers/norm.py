from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float, elementwise_affine: bool = True) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self._size = size

        if elementwise_affine:
            self.weight = torch.empty(size)
        # When elementwise_affine=False, we use a ones buffer created lazily
        # to ensure correct device/dtype

        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm
        self._ones_buffer: torch.Tensor | None = None

    def _get_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.elementwise_affine:
            return self.weight
        # Use cached ones buffer, recreate if needed for device/dtype match
        if self._ones_buffer is None or self._ones_buffer.device != x.device:
            self._ones_buffer = torch.ones(self._size, device=x.device, dtype=x.dtype)
        return self._ones_buffer

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self._get_weight(x)
        if residual is None:
            return self.rmsnorm(x, weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, weight, self.eps)
        return x, residual
