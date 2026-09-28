from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """Single 2D ConvLSTM cell used by DragNet."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so spatial dimensions are preserved.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = state
        combined = torch.cat((input_tensor, h_cur), dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        shape = (batch_size, self.hidden_dim, height, width)
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM used for DragNet's deterministic recurrent state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16),
        kernel_sizes: Sequence[int] = (3, 3),
        bias: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(kernel_sizes):
            raise ValueError("hidden_dims and kernel_sizes must have the same length.")
        if not hidden_dims:
            raise ValueError("At least one ConvLSTM layer is required.")

        cells: list[ConvLSTMCell] = []
        pairs = zip(hidden_dims, kernel_sizes, strict=True)
        for index, (hidden_dim, kernel_size) in enumerate(pairs):
            current_input_dim = input_dim if index == 0 else hidden_dims[index - 1]
            cells.append(
                ConvLSTMCell(
                    input_dim=current_input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
        self.cells = nn.ModuleList(cells)

    def _cell(self, index: int) -> ConvLSTMCell:
        return cast(ConvLSTMCell, self.cells[index])

    @property
    def output_dim(self) -> int:
        return self._cell(-1).hidden_dim

    def init_hidden(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            self._cell(index).init_hidden(
                batch_size,
                spatial_size,
                device=device,
                dtype=dtype,
            )
            for index in range(len(self.cells))
        ]

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if len(hidden_state) != len(self.cells):
            raise ValueError("hidden_state does not match the number of ConvLSTM layers.")

        new_hidden_state: list[tuple[torch.Tensor, torch.Tensor]] = []
        current = input_tensor
        for index, state in enumerate(hidden_state):
            cell = self._cell(index)
            h_next, c_next = cell(current, state)
            new_hidden_state.append((h_next, c_next))
            current = h_next
        return current, new_hidden_state
