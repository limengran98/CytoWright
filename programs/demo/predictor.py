"""Compact control-context predictor used by the feedback walkthrough."""
import torch
from torch import nn


def candidate_metadata():
    return {"schema_version": "cytowright_open_candidate_v1", "candidate_name": "control_context_predictor",
            "semantic_components": ["input.control_encoder", "response.readout", "objective.mse", "optimization.adam"],
            "component_symbols": {"input.control_encoder": ["ControlContextPredictor"],
                                  "response.readout": ["ControlContextPredictor"],
                                  "objective.mse": ["compute_loss"], "optimization.adam": ["build_optimizer"]},
            "change_summary": "Encode control context and predict the joint response."}


class ControlContextPredictor(nn.Module):
    def __init__(self, task_spec):
        super().__init__()
        self.control_encoder = nn.Sequential(nn.Linear(int(task_spec["context_dim"]), 32), nn.Tanh())
        self.response = nn.Linear(32, int(task_spec["target_dim"]))

    def forward(self, pre, chemical, dose):
        return 0.1*torch.tanh(self.response(self.control_encoder(pre)))


def build_model(task_spec):
    return ControlContextPredictor(task_spec)


def compute_loss(prediction, target, state):
    return torch.nn.functional.mse_loss(prediction, target)


def build_optimizer(model, state):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
