"""Compound-query attention for joint cellular-response prediction."""

import math
import torch
from torch import nn


def candidate_metadata():
    return {
        "schema_version": "cytowright_open_candidate_v1",
        "candidate_name": "bbbc047_response_predictor",
        "semantic_components": [
            "input.control_profile_encoder",
            "perturbation.chemical_encoder",
            "perturbation.dose_encoder",
            "perturbation.chemical_query_control_attention",
            "response.shared_program",
            "response.joint_readout",
            "objective.block_weighted_mse",
            "optimization.adamw",
            "optimization.cosine_schedule",
        ],
        "component_symbols": {
            "input.control_profile_encoder": ["ControlProfileEncoder"],
            "perturbation.chemical_encoder": ["ChemicalEncoder"],
            "perturbation.dose_encoder": ["DoseEncoder"],
            "perturbation.chemical_query_control_attention": [
                "ChemicalQueryControlAttention"
            ],
            "response.shared_program": ["SharedResponseProgram"],
            "response.joint_readout": ["JointResponseReadout"],
            "objective.block_weighted_mse": ["compute_loss"],
            "optimization.adamw": ["build_optimizer"],
            "optimization.cosine_schedule": ["build_scheduler"],
        },
        "change_summary": (
            "Separate control, compound and dose "
            "encoders, chemical-query attention over the control profile, dose "
            "residual with layer normalization, a shared response program, a "
            "block-aware CP/L1000 MSE objective, AdamW, and cosine scheduling."
        ),
    }


class ControlProfileEncoder(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(context_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pre):
        return self.norm(torch.nn.functional.gelu(self.proj(pre)))


class ChemicalEncoder(nn.Module):
    def __init__(self, chemical_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(chemical_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, chemical):
        return self.norm(torch.nn.functional.gelu(self.proj(chemical)))


class DoseEncoder(nn.Module):
    def __init__(self, dose_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dose_dim, hidden_dim)

    def forward(self, dose):
        if dose.dim() == 1:
            dose = dose.unsqueeze(-1)
        return torch.nn.functional.gelu(self.proj(dose))


class ChemicalQueryControlAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, control, chemical, dose):
        query = chemical.unsqueeze(1)
        key_value = control.unsqueeze(1)
        attn_out, _ = self.attn(
            query, key_value, key_value, need_weights=False
        )
        fused = self.norm(attn_out + dose.unsqueeze(1))
        return fused.squeeze(1)


class SharedResponseProgram(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, fused):
        return self.network(fused)


class JointResponseReadout(nn.Module):
    def __init__(self, hidden_dim, target_dim):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, target_dim)

    def forward(self, hidden):
        return self.projection(hidden)


class CrossModalAttentionModel(nn.Module):
    def __init__(self, task_spec):
        super().__init__()
        context_dim = int(task_spec["context_dim"])
        chemical_dim = int(task_spec["chemical_dim"])
        dose_dim = int(task_spec["dose_dim"])
        target_dim = int(task_spec["target_dim"])
        hidden_dim = 256
        self.control_encoder = ControlProfileEncoder(context_dim, hidden_dim)
        self.chemical_encoder = ChemicalEncoder(chemical_dim, hidden_dim)
        self.dose_encoder = DoseEncoder(dose_dim, hidden_dim)
        self.fusion = ChemicalQueryControlAttention(hidden_dim)
        self.response_program = SharedResponseProgram(hidden_dim)
        self.readout = JointResponseReadout(hidden_dim, target_dim)

    def forward(self, pre, chemical, dose):
        control = self.control_encoder(pre)
        chem = self.chemical_encoder(chemical)
        dose_embed = self.dose_encoder(dose)
        fused = self.fusion(control, chem, dose_embed)
        hidden = self.response_program(fused)
        return self.readout(hidden)


def build_model(task_spec):
    return CrossModalAttentionModel(task_spec)


def compute_loss(prediction, target, state):
    cp_dim = 701
    if isinstance(state, dict) and "cp_dim" in state:
        cp_dim = int(state["cp_dim"])
    cp_pred = prediction[:, :cp_dim]
    cp_tgt = target[:, :cp_dim]
    l1000_pred = prediction[:, cp_dim:]
    l1000_tgt = target[:, cp_dim:]
    cp_loss = torch.nn.functional.mse_loss(cp_pred, cp_tgt)
    l1000_loss = torch.nn.functional.mse_loss(l1000_pred, l1000_tgt)
    return 0.4 * cp_loss + 0.6 * l1000_loss


def build_optimizer(model, state):
    del state
    return torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-2
    )


def build_scheduler(optimizer, state):
    max_epochs = 150
    if isinstance(state, dict) and "max_epochs" in state:
        max_epochs = int(state["max_epochs"])
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )
