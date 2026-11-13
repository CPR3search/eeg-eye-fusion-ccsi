import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grl(x, lambda_):
    return _GradientReversal.apply(x, lambda_)


class SelectiveScanBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.delta_proj = nn.Linear(d_model, d_model)
        self.decay = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, T, D = x.shape
        u = self.in_proj(x)
        delta = torch.sigmoid(self.delta_proj(x))
        decay = torch.sigmoid(self.decay).view(1, D)
        h = torch.zeros(B, D, device=x.device)
        outs = []
        for t in range(T):
            h_new = decay * h * (1 - delta[:, t]) + delta[:, t] * u[:, t]
            m = mask[:, t].unsqueeze(1)
            h = m * h_new + (1 - m) * h
            outs.append(h)
        return self.out_proj(torch.stack(outs, dim=1))


def masked_mean(h, mask):
    m = mask.unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


class FeatureFusionModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_classes: int = 3, n_layers: int = 1):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.full((input_dim,), 0.5413))
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU())
        self.scan_fwd_layers = nn.ModuleList([SelectiveScanBlock(d_model) for _ in range(n_layers)])
        self.label_classifier = nn.Linear(d_model, n_classes)
        self.domain_classifier = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x, mask, lambda_=0.0):
        h = self.input_proj(x * F.softplus(self.feature_weight))
        for i, layer in enumerate(self.scan_fwd_layers):
            h = layer(h, mask) if i == 0 else h + layer(h, mask)
        pooled = masked_mean(h, mask)
        return self.label_classifier(pooled), self.domain_classifier(grl(pooled, lambda_)), pooled


class _Branch(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.full((in_dim,), 0.5413))
        self.input_proj = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU())
        self.scan = SelectiveScanBlock(d)

    def forward(self, x, mask):
        return masked_mean(self.scan(self.input_proj(x * F.softplus(self.feature_weight)), mask), mask)


class EncoderFusionModel(nn.Module):
    def __init__(self, input_dim: int, d_branch: int, eeg_dim: int, n_classes: int = 3):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.eeg = _Branch(eeg_dim, d_branch)
        self.eye = _Branch(input_dim - eeg_dim, d_branch)
        self.label_classifier = nn.Linear(2 * d_branch, n_classes)
        self.domain_classifier = nn.Sequential(nn.Linear(2 * d_branch, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x, mask, lambda_=0.0):
        pooled = torch.cat([self.eeg(x[..., :self.eeg_dim], mask), self.eye(x[..., self.eeg_dim:], mask)], dim=-1)
        return self.label_classifier(pooled), self.domain_classifier(grl(pooled, lambda_)), pooled


def _scan_params(d: int) -> int:
    return 3 * d * d + 4 * d


def _branch_params(in_dim: int, d: int) -> int:
    return in_dim + in_dim * d + d + _scan_params(d)


def _head_params(h: int, n_classes: int = 3) -> int:
    return h * n_classes + n_classes + h * 32 + 32 + 32 * 2 + 2


def feature_params(input_dim: int, d_model: int, n_layers: int = 1) -> int:
    return _branch_params(input_dim, d_model) + (n_layers - 1) * _scan_params(d_model) + _head_params(d_model)


def encoder_params(input_dim: int, d_branch: int, eeg_dim: int) -> int:
    return _branch_params(eeg_dim, d_branch) + _branch_params(input_dim - eeg_dim, d_branch) + _head_params(2 * d_branch)


def matched_branch_width(input_dim: int, eeg_dim: int, d_model: int) -> int:
    target = feature_params(input_dim, d_model)
    gaps = [(abs(encoder_params(input_dim, w, eeg_dim) - target) / target, w) for w in range(4, 513)]
    return min(gaps)[1]


def make_model(arch: str, input_dim: int, d_model: int, eeg_dim: int, d_branch: int | None = None) -> nn.Module:
    if arch == "feature":
        return FeatureFusionModel(input_dim, d_model)
    return EncoderFusionModel(input_dim, d_branch, eeg_dim)


def load_model(path, device: str) -> nn.Module:
    ck = torch.load(path, map_location=device)
    model = make_model(ck["arch"], ck["input_dim"], ck["d_model"], ck["eeg_dim"], ck.get("d_branch"))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


class RandomFourierFeatures(nn.Module):
    def __init__(self, d_in: int, D: int = 256, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("omega", torch.randn(d_in, D) / sigma)
        self.register_buffer("b", torch.rand(D) * 2 * np.pi)
        self.D = D

    def forward(self, x):
        return np.sqrt(2.0 / self.D) * torch.cos(x @ self.omega + self.b)


def median_sigma(z: torch.Tensor) -> float:
    with torch.no_grad():
        d = torch.cdist(z, z)
        med = d[d > 0].median().item()
    return max(med, 1e-3)
