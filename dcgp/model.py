import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-LN residual
        return x + self.block(self.norm(x))

# SEBlock
# reduction：16→8
# Ensure a minimum of 64 dimensions
class SEBlock(nn.Module):
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 64)),
            nn.GELU(),
            nn.Linear(max(dim // reduction, 64), dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


# Two layers, shallow wide network + stronger regularization
class dcgp_class(nn.Module):
    def __init__(self, input_dim: int = 4096, dropout: float = 0.3):
        super().__init__()

        h1 = 1024
        h2 = 512

        self.input_norm = nn.LayerNorm(input_dim)

        # Phase 1: Feature Extraction
        self.stage1 = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Phase 2: Feature Refinement (using only 1 residual block)
        self.res1 = ResidualBlock(h1, dropout=dropout)
        self.se1 = SEBlock(h1)

        # Phase 3: Dimensionality Reduction
        self.stage2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # output
        self.head = nn.Linear(h2, 2)

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.stage1(x)
        x = self.se1(self.res1(x))
        x = self.stage2(x)
        return self.head(x)
