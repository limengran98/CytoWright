"""Dose-scaled gated fusion for joint cellular-response prediction."""
import torch
from torch import nn


def candidate_metadata():
    return {
        'schema_version': 'cytowright_open_candidate_v1',
        'candidate_name': 'bbbc036_control_anchored_lowrank_dose_scaled',
        'semantic_components': [
            'input.separate_modality_encoders',
            'perturbation.control_anchored_gated_residual',
            'perturbation.softplus_dose_scaler',
            'response.control_baseline_plus_lowrank_residual',
            'objective.block_weighted_correlation_mse',
            'optimization.adamw',
            'optimization.cosine_schedule',
            'nuisance.layer_norm_dropout_residuals',
            'reliability.cp_l1000_block_weighting',
        ],
        'component_symbols': {
            'input.separate_modality_encoders': ['ControlEncoder', 'ChemicalEncoder'],
            'perturbation.control_anchored_gated_residual': ['ControlAnchoredFusion'],
            'perturbation.softplus_dose_scaler': ['DoseScaler'],
            'response.control_baseline_plus_lowrank_residual': ['LowRankResidualReadout'],
            'objective.block_weighted_correlation_mse': ['compute_loss'],
            'optimization.adamw': ['build_optimizer'],
            'optimization.cosine_schedule': ['build_scheduler'],
            'nuisance.layer_norm_dropout_residuals': ['ControlEncoder', 'ResidualBlock'],
            'reliability.cp_l1000_block_weighting': ['compute_loss'],
        },
        'change_summary': 'Dose-scaled gated fusion with a control baseline, a rank-64 perturbation readout, block-weighted response losses, AdamW and cosine scheduling.',
    }


class ControlEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, pre):
        return self.net(pre)


class ChemicalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, chemical):
        return self.net(chemical)


class DoseScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, dose):
        if dose.dim() == 1:
            dose = dose.unsqueeze(-1)
        return torch.nn.functional.softplus(dose * self.weight + self.bias) + 0.1


class ControlAnchoredFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, control, chemical, scale):
        gate = torch.sigmoid(self.gate(torch.cat([control, chemical], dim=-1)))
        perturb = scale * chemical
        return self.norm(control + gate * perturb)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class LowRankResidualReadout(nn.Module):
    def __init__(self, hidden_dim, target_dim, rank=64):
        super().__init__()
        self.baseline = nn.Linear(hidden_dim, target_dim)
        self.down = nn.Linear(hidden_dim, rank)
        self.up = nn.Linear(rank, target_dim)
        nn.init.normal_(self.baseline.weight, std=0.001)
        nn.init.zeros_(self.baseline.bias)
        nn.init.zeros_(self.up.bias)
        nn.init.normal_(self.up.weight, std=0.001)

    def forward(self, control, hidden):
        base = self.baseline(control)
        residual = self.up(torch.nn.functional.gelu(self.down(hidden)))
        return base + residual


class ControlAnchoredLowRankResponse(nn.Module):
    def __init__(self, task_spec):
        super().__init__()
        hidden_dim = 256
        self.control_encoder = ControlEncoder(int(task_spec['context_dim']), hidden_dim)
        self.chemical_encoder = ChemicalEncoder(int(task_spec['chemical_dim']), hidden_dim)
        self.dose_scaler = DoseScaler()
        self.fusion = ControlAnchoredFusion(hidden_dim)
        self.trunk = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.readout = LowRankResidualReadout(hidden_dim, int(task_spec['target_dim']))

    def forward(self, pre, chemical, dose):
        control = self.control_encoder(pre)
        chemical_state = self.chemical_encoder(chemical)
        scale = self.dose_scaler(dose)
        hidden = self.fusion(control, chemical_state, scale)
        hidden = self.trunk(hidden)
        return self.readout(control, hidden)


def build_model(task_spec):
    return ControlAnchoredLowRankResponse(task_spec)


def _block_correlation(prediction, target):
    pred_centered = prediction - prediction.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=0) * (target_centered ** 2).sum(dim=0)
    ) + 1e-8
    return (numerator / denominator).mean()


def compute_loss(prediction, target, state):
    cp_dim = 591
    if isinstance(state, dict):
        cp_dim = int(state.get('cp_dim', cp_dim))
    cp_pred = prediction[:, :cp_dim]
    cp_target = target[:, :cp_dim]
    l1000_pred = prediction[:, cp_dim:]
    l1000_target = target[:, cp_dim:]
    cp_mse = torch.nn.functional.mse_loss(cp_pred, cp_target)
    l1000_mse = torch.nn.functional.mse_loss(l1000_pred, l1000_target)
    cp_corr = _block_correlation(cp_pred, cp_target)
    l1000_corr = _block_correlation(l1000_pred, l1000_target)
    cp_loss = cp_mse + 0.1 * (1.0 - cp_corr)
    l1000_loss = l1000_mse + 0.1 * (1.0 - l1000_corr)
    return 0.4 * cp_loss + 0.6 * l1000_loss


def build_optimizer(model, state):
    del state
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)


def build_scheduler(optimizer, state):
    max_epochs = 150
    if isinstance(state, dict):
        max_epochs = int(state.get('max_epochs', max_epochs))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
