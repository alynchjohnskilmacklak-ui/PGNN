"""Neural network architecture for ballistic angle prediction."""

import torch
import torch.nn as nn

from ballistics import ALPHA_ABS_MAX

LOW_THETA_MIN = 0.0
LOW_THETA_MAX = 55.0
HIGH_THETA_MIN = 45.0
HIGH_THETA_MAX = 85.0


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.12):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AngleHead(nn.Module):
    def __init__(self, hidden: int, theta_min: float, theta_max: float, alpha_abs_max: float):
        super().__init__()
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.alpha_abs_max = float(alpha_abs_max)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 2),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(feats)
        theta = torch.sigmoid(raw[:, 0:1]) * (self.theta_max - self.theta_min) + self.theta_min
        alpha = torch.tanh(raw[:, 1:2]) * self.alpha_abs_max
        return torch.cat([theta, alpha], dim=1)


class SingleBranchDNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 14,
        hidden: int = 192,
        num_blocks: int = 3,
        dropout: float = 0.12,
        theta_min: float = LOW_THETA_MIN,
        theta_max: float = LOW_THETA_MAX,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.backbone = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(num_blocks)])
        self.head = AngleHead(hidden, theta_min, theta_max, ALPHA_ABS_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(self.stem(x))
        return self.head(feats)


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_range: tuple[float, float] = (-0.1, 1.1),
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)

        grid_min, grid_max = float(grid_range[0]), float(grid_range[1])
        h = (grid_max - grid_min) / self.grid_size
        grid = (
            torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=torch.float32)
            * h
            + grid_min
        )
        self.register_buffer("grid", grid.expand(self.in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, self.grid_size + self.spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.spline_scaler, a=5 ** 0.5)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :-(k + 1)]
            left_den = grid[:, k:-1] - grid[:, :-(k + 1)]
            right_num = grid[:, k + 1:] - x
            right_den = grid[:, k + 1:] - grid[:, 1:-k]
            bases = (
                left_num / torch.clamp(left_den, min=1e-8) * bases[:, :, :-1]
                + right_num / torch.clamp(right_den, min=1e-8) * bases[:, :, 1:]
            )
        return bases.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = torch.nn.functional.linear(self.base_activation(x), self.base_weight)
        spline_basis = self.b_splines(x).reshape(x.shape[0], -1)
        spline_weight = self.scaled_spline_weight.reshape(self.out_features, -1)
        spline_output = torch.nn.functional.linear(spline_basis, spline_weight)
        return (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)


class SingleBranchKAN(nn.Module):
    def __init__(
        self,
        in_dim: int = 14,
        hidden: int = 128,
        num_layers: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
        theta_min: float = LOW_THETA_MIN,
        theta_max: float = LOW_THETA_MAX,
    ):
        super().__init__()
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.alpha_abs_max = float(ALPHA_ABS_MAX)

        layers = [in_dim] + [hidden] * int(max(1, num_layers - 1)) + [2]
        self.layers = nn.ModuleList(
            KANLinear(
                layers[i],
                layers[i + 1],
                grid_size=grid_size,
                spline_order=spline_order,
            )
            for i in range(len(layers) - 1)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(width) for width in layers[1:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.norms[idx](x)
        theta = torch.sigmoid(x[:, 0:1]) * (self.theta_max - self.theta_min) + self.theta_min
        alpha = torch.tanh(x[:, 1:2]) * self.alpha_abs_max
        return torch.cat([theta, alpha], dim=1)


class SingleBranchKANMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 14,
        hidden: int = 192,
        num_blocks: int = 3,
        dropout: float = 0.12,
        grid_size: int = 3,
        spline_order: int = 3,
        theta_min: float = LOW_THETA_MIN,
        theta_max: float = LOW_THETA_MAX,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            KANLinear(
                in_dim,
                hidden,
                grid_size=grid_size,
                spline_order=spline_order,
            ),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.backbone = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(num_blocks)])
        self.head = AngleHead(hidden, theta_min, theta_max, ALPHA_ABS_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(self.stem(x))
        return self.head(feats)


def build_single_branch_model(
    model_type: str,
    in_dim: int,
    hidden: int,
    dropout: float,
    theta_min: float,
    theta_max: float,
) -> nn.Module:
    model_type = str(model_type).lower()
    if model_type == "mlp":
        return SingleBranchDNN(
            in_dim=in_dim,
            hidden=hidden,
            dropout=dropout,
            theta_min=theta_min,
            theta_max=theta_max,
        )
    if model_type == "kan":
        return SingleBranchKAN(
            in_dim=in_dim,
            hidden=hidden,
            theta_min=theta_min,
            theta_max=theta_max,
        )
    if model_type in ("kan_mlp", "hybrid", "hybrid_kan"):
        return SingleBranchKANMLP(
            in_dim=in_dim,
            hidden=hidden,
            dropout=dropout,
            theta_min=theta_min,
            theta_max=theta_max,
        )
    raise ValueError(f"Unknown model_type: {model_type!r}. Expected 'mlp', 'kan', or 'kan_mlp'.")


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = None
        for name, value in model.state_dict().items():
            self.shadow[name] = value.detach().clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[name].copy_(value.detach())

    def apply_to(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None
