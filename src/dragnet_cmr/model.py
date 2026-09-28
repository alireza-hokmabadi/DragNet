from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch import nn

from .convlstm import ConvLSTM
from .losses import LossTerms, dragnet_loss_terms
from .spatial import warp_image_2d

DisplacementSampling = Literal["legacy", "cholesky"]


class DragNetForward(NamedTuple):
    registered: torch.Tensor
    displacement: torch.Tensor
    losses: LossTerms


class DragNetGeneration(NamedTuple):
    generated: torch.Tensor
    displacement: torch.Tensor


def _conv_down(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
    )


def _conv_up(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
    )


class DragNet(nn.Module):
    """Deformable Registration and Generative Network from Zakeri et al. (2023).

    The convolutional dimensions are intentionally kept compatible with the published
    128 x 128 implementation. Engineering around the network has been modernised, but
    the core module layout and loss weights mirror the historical public code.
    """

    image_size = (128, 128)
    feature_size = (32, 32)
    latent_dim = 64

    def __init__(
        self,
        *,
        displacement_sampling: DisplacementSampling = "legacy",
        latent_kl_weight: float = 2e-4,
        smoothness_weight: float = 0.03,
        displacement_kl_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        if displacement_sampling not in {"legacy", "cholesky"}:
            raise ValueError("displacement_sampling must be 'legacy' or 'cholesky'.")
        self.displacement_sampling = displacement_sampling
        self.latent_kl_weight = latent_kl_weight
        self.smoothness_weight = smoothness_weight
        self.displacement_kl_weight = displacement_kl_weight
        self.activation = nn.LeakyReLU(0.2)

        # Prior p(z_t | h_{t-1})
        self.conv_z_prior = _conv_down(16, 4)
        self.fc_mu_z_prior = nn.Linear(4 * 16 * 16, self.latent_dim)
        self.fc_logvar_z_prior = nn.Linear(4 * 16 * 16, self.latent_dim)

        # Image feature map phi_x
        self.conv1_phi_x = _conv_down(1, 32)
        self.conv2_phi_x = _conv_down(32, 32)

        # Approximate posterior q(z_t | I_t, h_{t-1})
        self.conv_infer = _conv_down(32 + 16, 16)
        self.fc_mu_infer = nn.Linear(16 * 16 * 16, self.latent_dim)
        self.fc_logvar_infer = nn.Linear(16 * 16 * 16, self.latent_dim)

        # Latent feature map phi_z
        self.fc_phi_z = nn.Linear(self.latent_dim, 16 * 16 * 16)
        self.conv_phi_z = _conv_up(16, 32)

        # Displacement posterior q(D_t | I_{t-1}, z_t)
        self.conv1_gen = _conv_up(32 + 32, 32)
        self.conv2_gen = _conv_up(32, 32)
        self.conv_mu_gen = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.conv_logvar_gen = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.conv_v_gen = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

        self.recurrent = ConvLSTM(
            input_dim=32 + 32,
            hidden_dims=(32, 16),
            kernel_sizes=(3, 3),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_sequence(self, sequence: torch.Tensor) -> None:
        if sequence.ndim != 5:
            raise ValueError("sequence must have shape (B, T, 1, 128, 128).")
        if sequence.shape[2] != 1:
            raise ValueError("DragNet expects one image channel.")
        if tuple(sequence.shape[-2:]) != self.image_size:
            raise ValueError(f"DragNet expects spatial size {self.image_size}.")
        if sequence.shape[1] < 2:
            raise ValueError("At least two cardiac phases are required.")

    def _validate_frame(self, frame: torch.Tensor) -> None:
        if frame.ndim != 4 or frame.shape[1] != 1:
            raise ValueError("frame must have shape (B, 1, 128, 128).")
        if tuple(frame.shape[-2:]) != self.image_size:
            raise ValueError(f"DragNet expects spatial size {self.image_size}.")

    def _initial_recurrent_state(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size = reference.shape[0]
        height, width = self.feature_size
        h_recurrent = torch.zeros(
            (batch_size, 16, height, width),
            device=reference.device,
            dtype=reference.dtype,
        )
        hidden_state = self.recurrent.init_hidden(
            batch_size,
            self.feature_size,
            device=reference.device,
            dtype=reference.dtype,
        )
        return h_recurrent, hidden_state

    def z_prior(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.activation(self.conv_z_prior(hidden))
        value = value.reshape(value.shape[0], -1)
        return self.fc_mu_z_prior(value), self.fc_logvar_z_prior(value)

    def image_features(self, image: torch.Tensor) -> torch.Tensor:
        value = self.activation(self.conv1_phi_x(image))
        return self.activation(self.conv2_phi_x(value))

    def infer_z(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.activation(self.conv_infer(features))
        value = value.reshape(value.shape[0], -1)
        return self.fc_mu_infer(value), self.fc_logvar_infer(value)

    @staticmethod
    def sample_diagonal_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def displacement_covariance(log_var: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
        vector = torch.exp(log_v).permute(0, 2, 3, 1).unsqueeze(-1)
        covariance = torch.matmul(vector, vector.transpose(-1, -2))
        diagonal = torch.exp(log_var[:, 0]) + 1e-6
        covariance[..., 0, 0] = covariance[..., 0, 0] + diagonal
        covariance[..., 1, 1] = covariance[..., 1, 1] + diagonal
        return covariance

    def sample_displacement(
        self,
        mu: torch.Tensor,
        covariance: torch.Tensor,
    ) -> torch.Tensor:
        mu_matrix = mu.permute(0, 2, 3, 1).unsqueeze(-1)
        epsilon = torch.randn_like(mu_matrix)
        if self.displacement_sampling == "legacy":
            # Preserves the historical public implementation exactly.
            transform = 0.5 * covariance
        else:
            identity = torch.eye(2, device=covariance.device, dtype=covariance.dtype)
            transform = torch.linalg.cholesky(covariance + 1e-6 * identity)
        sample = mu_matrix + torch.matmul(transform, epsilon)
        return sample.squeeze(-1).permute(0, 3, 1, 2)

    def latent_features(self, z_value: torch.Tensor) -> torch.Tensor:
        value = self.fc_phi_z(z_value).reshape(-1, 16, 16, 16)
        return self.activation(self.conv_phi_z(value))

    def infer_displacement(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.activation(self.conv1_gen(features))
        value = self.activation(self.conv2_gen(value))
        return self.conv_mu_gen(value), self.conv_logvar_gen(value), self.conv_v_gen(value)

    def _update_recurrent(
        self,
        z_features: torch.Tensor,
        image_features: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        recurrent_input = torch.cat((z_features, image_features), dim=1)
        return self.recurrent(recurrent_input, hidden_state)

    def forward(self, sequence: torch.Tensor) -> DragNetForward:
        self._validate_sequence(sequence)
        batch_size, frame_count, _, height, width = sequence.shape
        h_recurrent, hidden_state = self._initial_recurrent_state(sequence[:, 0])

        registered = torch.zeros(
            (batch_size, frame_count, 1, height, width),
            device=sequence.device,
            dtype=sequence.dtype,
        )
        displacement = torch.zeros(
            (batch_size, frame_count, 2, height, width),
            device=sequence.device,
            dtype=sequence.dtype,
        )

        similarity = sequence.new_zeros(())
        latent_kl = sequence.new_zeros(())
        smoothness = sequence.new_zeros(())
        displacement_kl = sequence.new_zeros(())

        for time_index in range(frame_count + 1):
            current_index = time_index % frame_count
            past_index = (time_index - 1) % frame_count
            current = sequence[:, current_index]
            past = sequence[:, past_index]

            z_prior_mu, z_prior_logvar = self.z_prior(h_recurrent)
            current_features = self.image_features(current)
            z_mu, z_logvar = self.infer_z(torch.cat((current_features, h_recurrent), dim=1))
            z_value = self.sample_diagonal_gaussian(z_mu, z_logvar)
            z_features = self.latent_features(z_value)

            past_features = self.image_features(past)
            d_mu, d_logvar, d_log_v = self.infer_displacement(
                torch.cat((z_features, past_features), dim=1)
            )
            d_cov = self.displacement_covariance(d_logvar, d_log_v)
            d_value = self.sample_displacement(d_mu, d_cov)

            # The historical implementation stored D as (row, column) and swapped
            # channels before the STN. warp_image_2d expects (x, y), so preserve that.
            prediction = warp_image_2d(past, d_value[:, [1, 0]])
            h_recurrent, hidden_state = self._update_recurrent(
                z_features,
                current_features,
                hidden_state,
            )

            if time_index > 0:
                registered[:, current_index] = prediction
                displacement[:, current_index] = d_value
                terms = dragnet_loss_terms(
                    current,
                    prediction,
                    z_mu,
                    z_logvar,
                    z_prior_mu,
                    z_prior_logvar,
                    d_value,
                    d_mu,
                    d_cov,
                    latent_kl_weight=self.latent_kl_weight,
                    smoothness_weight=self.smoothness_weight,
                    displacement_kl_weight=self.displacement_kl_weight,
                )
                similarity = similarity + terms.similarity
                latent_kl = latent_kl + terms.latent_kl
                smoothness = smoothness + terms.smoothness
                displacement_kl = displacement_kl + terms.displacement_kl

        losses = LossTerms(
            similarity=similarity,
            latent_kl=latent_kl,
            smoothness=smoothness,
            displacement_kl=displacement_kl,
            total=similarity + latent_kl + smoothness + displacement_kl,
        )
        return DragNetForward(registered, displacement, losses)

    @torch.no_grad()
    def generate_from_one_frame(
        self,
        first_frame: torch.Tensor,
        frame_count: int = 7,
    ) -> DragNetGeneration:
        self._validate_frame(first_frame)
        if frame_count < 2:
            raise ValueError("frame_count must be at least 2.")
        batch_size, _, height, width = first_frame.shape
        h_recurrent, hidden_state = self._initial_recurrent_state(first_frame)
        generated = torch.zeros(
            (batch_size, frame_count, 1, height, width),
            device=first_frame.device,
            dtype=first_frame.dtype,
        )
        displacement = torch.zeros(
            (batch_size, frame_count, 2, height, width),
            device=first_frame.device,
            dtype=first_frame.dtype,
        )

        previous = first_frame.clone()
        image_features = self.image_features(first_frame)
        z_mu, z_logvar = self.infer_z(torch.cat((image_features, h_recurrent), dim=1))
        z_value = self.sample_diagonal_gaussian(z_mu, z_logvar)
        z_features = self.latent_features(z_value)
        h_recurrent, hidden_state = self._update_recurrent(z_features, image_features, hidden_state)

        for time_index in range(1, frame_count + 1):
            current_index = time_index % frame_count
            z_prior_mu, z_prior_logvar = self.z_prior(h_recurrent)
            z_value = self.sample_diagonal_gaussian(z_prior_mu, z_prior_logvar)
            z_features = self.latent_features(z_value)
            previous_features = self.image_features(previous)
            d_mu, d_logvar, d_log_v = self.infer_displacement(
                torch.cat((z_features, previous_features), dim=1)
            )
            d_cov = self.displacement_covariance(d_logvar, d_log_v)
            d_value = self.sample_displacement(d_mu, d_cov)
            output = warp_image_2d(previous, d_value[:, [1, 0]])

            generated[:, current_index] = output
            displacement[:, current_index] = d_value
            image_features = self.image_features(output)
            previous = output.clone()
            h_recurrent, hidden_state = self._update_recurrent(
                z_features,
                image_features,
                hidden_state,
            )
        return DragNetGeneration(generated, displacement)

    @torch.no_grad()
    def generate_from_two_frames(
        self,
        first_frame: torch.Tensor,
        second_frame: torch.Tensor,
        frame_count: int = 7,
    ) -> DragNetGeneration:
        self._validate_frame(first_frame)
        self._validate_frame(second_frame)
        if first_frame.shape != second_frame.shape:
            raise ValueError("first_frame and second_frame must have identical shapes.")
        if frame_count < 2:
            raise ValueError("frame_count must be at least 2.")

        batch_size, _, height, width = first_frame.shape
        h_recurrent, hidden_state = self._initial_recurrent_state(first_frame)
        generated = torch.zeros(
            (batch_size, frame_count, 1, height, width),
            device=first_frame.device,
            dtype=first_frame.dtype,
        )
        displacement = torch.zeros(
            (batch_size, frame_count, 2, height, width),
            device=first_frame.device,
            dtype=first_frame.dtype,
        )

        first_features = self.image_features(first_frame)
        z_mu, z_logvar = self.infer_z(torch.cat((first_features, h_recurrent), dim=1))
        z_value = self.sample_diagonal_gaussian(z_mu, z_logvar)
        z_features = self.latent_features(z_value)
        h_recurrent, hidden_state = self._update_recurrent(z_features, first_features, hidden_state)
        previous = first_frame.clone()

        for time_index in range(1, frame_count + 1):
            current_index = time_index % frame_count
            if time_index == 1:
                image_features = self.image_features(second_frame)
                z_mu, z_logvar = self.infer_z(torch.cat((image_features, h_recurrent), dim=1))
                z_value = self.sample_diagonal_gaussian(z_mu, z_logvar)
                z_features = self.latent_features(z_value)
            else:
                z_prior_mu, z_prior_logvar = self.z_prior(h_recurrent)
                z_value = self.sample_diagonal_gaussian(z_prior_mu, z_prior_logvar)
                z_features = self.latent_features(z_value)

            previous_features = self.image_features(previous)
            d_mu, d_logvar, d_log_v = self.infer_displacement(
                torch.cat((z_features, previous_features), dim=1)
            )
            d_cov = self.displacement_covariance(d_logvar, d_log_v)
            d_value = self.sample_displacement(d_mu, d_cov)
            output = warp_image_2d(previous, d_value[:, [1, 0]])
            generated[:, current_index] = output
            displacement[:, current_index] = d_value

            if time_index == 1:
                previous = second_frame.clone()
            else:
                previous = output.clone()
                image_features = self.image_features(output)

            h_recurrent, hidden_state = self._update_recurrent(
                z_features,
                image_features,
                hidden_state,
            )
        return DragNetGeneration(generated, displacement)
