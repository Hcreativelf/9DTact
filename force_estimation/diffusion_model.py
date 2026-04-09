"""
Diffusion Model for 6D Force Estimation from Tactile Images.

Architecture:
- CNN encoder (ResNet/DenseNet) extracts image features as condition
- A small MLP-based denoising network predicts noise in the 6D force space
- DDPM forward process adds noise to force vectors
- Reverse process iteratively denoises to predict force from image condition

The force (wrench) is the diffusion target, and the tactile image is the condition.
"""

import math
import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# 1. CNN Feature Encoder (extracts condition from tactile image)
# ============================================================

class ImageConditionEncoder(nn.Module):
    """
    Uses a pretrained CNN backbone to extract a feature vector from the tactile image.
    The final classification/regression head is removed; we use the penultimate features.
    """

    def __init__(self, backbone='Densenet', layer=169, pretrained=True, feature_dim=256):
        super().__init__()
        self.backbone_name = backbone

        if backbone == 'Resnet':
            if layer == 18:
                base = models.resnet18(pretrained=pretrained)
            elif layer == 34:
                base = models.resnet34(pretrained=pretrained)
            elif layer == 50:
                base = models.resnet50(pretrained=pretrained)
            else:
                base = models.resnet101(pretrained=pretrained)
            in_features = base.fc.in_features
            base.fc = nn.Identity()  # remove the original head
            self.encoder = base
        elif backbone == 'Densenet':
            if layer == 121:
                base = models.densenet121(pretrained=pretrained)
            elif layer == 161:
                base = models.densenet161(pretrained=pretrained)
            elif layer == 169:
                base = models.densenet169(pretrained=pretrained)
            else:
                base = models.densenet201(pretrained=pretrained)
            in_features = base.classifier.in_features
            base.classifier = nn.Identity()
            self.encoder = base
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Project to a fixed feature dimension
        self.proj = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.SiLU(),
        )
        self.out_dim = feature_dim
        print(f"[Encoder] {backbone}-{layer}, feature_dim={feature_dim}")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) tactile image tensor
        Returns:
            (B, feature_dim) condition vector
        """
        feat = self.encoder(x)  # (B, in_features)
        return self.proj(feat)   # (B, feature_dim)


# ============================================================
# 2. Sinusoidal Timestep Embedding
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Standard sinusoidal embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, dim) embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)    # 计算正弦位置编码中频率的衰减尺度
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)    # 计算频率向量
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


# ============================================================
# 3. Denoising MLP (predicts noise in force space)
# ============================================================

class DenoisingMLP(nn.Module):
    """
    A small MLP that predicts the noise added to the 6D force vector,
    conditioned on the image features and the current timestep.

    Input:  noisy force (6) + image condition (feature_dim) + timestep embedding (time_dim)
    Output: predicted noise (6)
    """

    def __init__(self, force_dim=6, cond_dim=256, time_dim=128, hidden_dim=512, num_layers=6):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Force embedding
        self.force_embed = nn.Sequential(
            nn.Linear(force_dim, hidden_dim),
            nn.SiLU(),
        )

        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
        )

        # Time embedding projection to hidden_dim
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )

        # Main residual blocks
        layers = []
        for i in range(num_layers):
            layers.append(ResidualBlock(hidden_dim))
        self.blocks = nn.ModuleList(layers)

        # Output head
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, force_dim),
        )

    def forward(self, x_noisy, t, cond):
        """
        Args:
            x_noisy: (B, 6) noisy force vector
            t:       (B,)   diffusion timestep
            cond:    (B, cond_dim) image condition features
        Returns:
            (B, 6) predicted noise
        """
        t_emb = self.time_embed(t)       # (B, time_dim)

        h_force = self.force_embed(x_noisy)  # (B, hidden_dim)
        h_cond = self.cond_embed(cond)       # (B, hidden_dim)
        h_time = self.time_proj(t_emb)       # (B, hidden_dim)

        # Combine: additive fusion
        h = h_force + h_cond + h_time  # (B, hidden_dim)

        for block in self.blocks:
            h = block(h)

        return self.out_head(h)  # (B, 6)


class ResidualBlock(nn.Module):
    """Simple residual block with LayerNorm."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


# ============================================================
# 4. Gaussian Diffusion Scheduler (DDPM)
# ============================================================

class GaussianDiffusion:
    """
    DDPM scheduler for the 6D force space.
    Manages the noise schedule, forward process (q), and reverse process (p).
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])

        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance: beta_tilde_t (clamp to avoid division by zero at t=0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-8)
        )
        # Clip the first element to prevent sqrt of negative
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_start: (B, 6) clean force
            t:       (B,)   timestep indices
            noise:   optional pre-generated noise
        Returns:
            x_noisy: (B, 6) noisy force at timestep t
            noise:   (B, 6) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)   # ϵ

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(-1)       # (B, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # (B, 1)

        x_noisy = sqrt_alpha_bar * x_start + sqrt_one_minus * noise
        return x_noisy, noise

    def p_sample(self, model_output, x_t, t):
        """
        Single reverse step: p(x_{t-1} | x_t)

        Args:
            model_output: (B, 6) predicted noise
            x_t:          (B, 6) current noisy sample
            t:            (B,)   current timestep
        Returns:
            x_prev: (B, 6) denoised one step
        """
        betas_t = self.betas[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        sqrt_recip = self.sqrt_recip_alphas[t].unsqueeze(-1)

        # Predicted mean
        pred_mean = sqrt_recip * (x_t - betas_t / sqrt_one_minus * model_output)

        if (t == 0).all():
            return pred_mean
        else:
            posterior_var = self.posterior_variance[t].unsqueeze(-1)
            noise = torch.randn_like(x_t)
            # For samples where t==0, don't add noise
            mask = (t > 0).float().unsqueeze(-1)
            return pred_mean + mask * torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, cond, num_samples=None, return_trajectory=False):
        """
        Full reverse diffusion: generate force prediction from pure noise.

        Args:
            denoise_fn: callable(x_noisy, t, cond) -> predicted_noise
            cond:       (B, cond_dim) image condition
            num_samples: if None, B is inferred from cond
            return_trajectory: if True, return all intermediate steps
        Returns:
            x_0: (B, 6) predicted clean force
        """
        batch_size = cond.shape[0]
        x = torch.randn(batch_size, 6, device=self.device)  # start from noise

        trajectory = [x] if return_trajectory else None

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise = denoise_fn(x, t, cond)
            x = self.p_sample(predicted_noise, x, t)
            if return_trajectory:
                trajectory.append(x)

        return (x, trajectory) if return_trajectory else x

    @torch.no_grad()
    def ddim_sample(self, denoise_fn, cond, ddim_steps=50, eta=0.0):
        """
        DDIM sampling for much faster inference.
        With eta=0, this is deterministic (no stochasticity).

        Args:
            denoise_fn: callable(x_noisy, t, cond) -> predicted_noise
            cond:       (B, cond_dim) image condition
            ddim_steps: number of sampling steps (e.g. 50 instead of 1000)
            eta:        stochasticity (0 = deterministic DDIM, 1 = DDPM)
        Returns:
            x_0: (B, 6) predicted clean force
        """
        batch_size = cond.shape[0]
        # Create sub-sequence of timesteps
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(batch_size, 6, device=self.device)

        for i, t_cur in enumerate(timesteps):
            t = torch.full((batch_size,), t_cur, device=self.device, dtype=torch.long)
            predicted_noise = denoise_fn(x, t, cond)

            alpha_bar_t = self.alphas_cumprod[t_cur]
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)

            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            # Compute sigma
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8) * (1 - alpha_bar_t / alpha_bar_prev)
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0)) * predicted_noise

            # Combine
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
            if sigma > 0 and t_cur > 0:
                x = x + sigma * torch.randn_like(x)

        return x


# ============================================================
# 5. Complete Diffusion Force Estimator
# ============================================================

class DiffusionForceEstimator(nn.Module):
    """
    Full model combining:
    - CNN image encoder (condition extractor)
    - MLP denoiser (noise predictor in 6D force space)

    The GaussianDiffusion scheduler is external and handles the forward/reverse process.
    """

    def __init__(self, backbone='Densenet', layer=169, pretrained=True,
                 feature_dim=256, time_dim=128, hidden_dim=512, num_mlp_layers=6):
        super().__init__()
        self.encoder = ImageConditionEncoder(
            backbone=backbone, layer=layer, pretrained=pretrained, feature_dim=feature_dim
        )
        self.denoiser = DenoisingMLP(
            force_dim=6,
            cond_dim=feature_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
        )

    def forward(self, x_noisy, t, images):
        """
        Forward pass: predict noise given noisy force, timestep, and image.

        Args:
            x_noisy: (B, 6) noisy force
            t:       (B,)   timestep
            images:  (B, 3, H, W) tactile image
        Returns:
            (B, 6) predicted noise
        """
        cond = self.encoder(images)
        return self.denoiser(x_noisy, t, cond)

    def predict_force(self, images, diffusion, num_ensemble=10, use_ddim=True, ddim_steps=50):
        """
        Inference: predict force by running reverse diffusion and averaging multiple samples.

        Args:
            images:       (B, 3, H, W) input images
            diffusion:    GaussianDiffusion instance
            num_ensemble: number of reverse passes to average (reduces variance)
            use_ddim:     if True, use DDIM (much faster); if False, use full DDPM
            ddim_steps:   number of DDIM steps (only used if use_ddim=True)
        Returns:
            (B, 6) averaged force prediction
        """
        self.eval()
        cond = self.encoder(images)  # (B, feature_dim)

        predictions = []
        for _ in range(num_ensemble):
            if use_ddim:
                pred = diffusion.ddim_sample(
                    denoise_fn=lambda x_noisy, t, c: self.denoiser(x_noisy, t, c),
                    cond=cond,
                    ddim_steps=ddim_steps,
                )
            else:
                pred = diffusion.p_sample_loop(
                    denoise_fn=lambda x_noisy, t, c: self.denoiser(x_noisy, t, c),
                    cond=cond,
                )
            predictions.append(pred)

        # Average over ensemble
        return torch.stack(predictions, dim=0).mean(dim=0)  # (B, 6)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionForceEstimator(backbone='Densenet', layer=169, pretrained=False).to(device)
    diffusion = GaussianDiffusion(num_timesteps=1000, device=device)

    # Test forward
    dummy_img = torch.randn(4, 3, 345, 460, device=device)
    dummy_force = torch.randn(4, 6, device=device)
    t = torch.randint(0, 1000, (4,), device=device)

    noise_pred = model(dummy_force, t, dummy_img)
    print(f"Noise prediction shape: {noise_pred.shape}")  # (4, 6)

    # Test inference
    with torch.no_grad():
        pred_force = model.predict_force(dummy_img, diffusion, num_ensemble=3)
        print(f"Predicted force shape: {pred_force.shape}")  # (4, 6)
