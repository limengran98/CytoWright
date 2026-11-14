"""Load fitted predictors, generate responses and measure compound contribution."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import types

import numpy as np
import torch

from .feedback.core import FeedbackModel, metrics, predict_raw
from .feedback.editing import Standardizer

NORMALIZER_FIELDS = ("pre_mean", "pre_scale", "condition_mean", "condition_scale", "target_mean", "target_scale")


def load_arrays(path, *, target=False):
    with np.load(path, allow_pickle=False) as archive:
        keys = ["pre", "condition"] + (["target"] if target else [])
        arrays = {key: np.asarray(archive[key], dtype=np.float32) for key in keys}
    rows = len(arrays["pre"])
    if any(value.ndim != 2 or len(value) != rows or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Input arrays must be finite, two-dimensional and row-aligned")
    if arrays["condition"].shape[1] != 2049:
        raise ValueError("condition must contain 2,048 compound features followed by dose")
    return arrays


def save_normalizer(normalizer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: getattr(normalizer, key).detach().cpu().numpy() for key in NORMALIZER_FIELDS})


def load_normalizer(path):
    result = Standardizer.__new__(Standardizer)
    fingerprint = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for key in NORMALIZER_FIELDS:
            value = np.asarray(archive[key], dtype=np.float32)
            setattr(result, key, torch.from_numpy(value.copy()))
            fingerprint.update(str(tuple(value.shape)).encode("ascii"))
            fingerprint.update(value.tobytes())
    result.fingerprint = fingerprint.hexdigest()
    return result


def load_predictor(source, checkpoint, normalizer):
    normalizer = load_normalizer(normalizer)
    context = int(normalizer.pre_mean.shape[1])
    target = int(normalizer.target_mean.shape[1])
    module = types.ModuleType("cytowright_predictor")
    source = Path(source)
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    model = module.build_model({"context_dim": context, "chemical_dim": 2048, "dose_dim": 1,
                                "target_dim": target, "cp_dim": context, "l1000_dim": target-context})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    if "chemical_basis" in state:
        spec = payload["context"]
        model = FeedbackModel(model, spec["mode"], spec["dimension"],
                              rank=state["chemical_basis"].shape[1], target_dim=target)
    recorded = payload.get("standardization_fingerprint", payload.get("normalizer_fingerprint"))
    if recorded and recorded != normalizer.fingerprint:
        raise ValueError("Checkpoint and normalization statistics must match")
    model.load_state_dict(state, strict=True)
    return model.cpu().eval(), SimpleNamespace(standardizer=normalizer)


def predict_file(source, checkpoint, normalizer, data, output):
    model, tensors = load_predictor(source, checkpoint, normalizer)
    arrays = load_arrays(data)
    prediction = predict_raw(model, tensors, arrays["pre"], arrays["condition"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, prediction)
    return {"status": "complete", "samples": len(prediction), "response_features": prediction.shape[1], "output": str(output)}


def replacement_groups(pre, condition):
    groups = {}
    for index in range(len(pre)):
        key = pre[index].tobytes() + condition[index, -1:].tobytes()
        groups.setdefault(key, []).append(index)
    eligible = {}
    for rows in groups.values():
        for index in rows:
            candidates = [other for other in rows if not np.array_equal(condition[index, :2048], condition[other, :2048])]
            if candidates:
                eligible[index] = candidates
    return eligible


def compound_analysis(model, tensors, arrays, *, maps=32, seed=41):
    if maps < 1:
        raise ValueError("maps must be positive")
    pre, condition, target = (arrays[key] for key in ["pre", "condition", "target"])
    prediction = predict_raw(model, tensors, pre, condition)
    scores = metrics(prediction, target, cp_dim=pre.shape[1])
    scores = {key: float(value) if np.isfinite(value) else None for key, value in scores.items()}
    eligible = replacement_groups(pre, condition)
    rows = np.asarray(sorted(eligible), dtype=int)
    result = {"samples": len(pre), "prediction": scores, "matched_samples": len(rows),
              "coverage": len(rows)/len(pre), "replacement_maps": maps, "seed": seed,
              "compound_mse_increase": None, "prediction_rms_change": None}
    if not len(rows):
        return result
    reference = prediction[rows].astype(np.float64)
    truth = target[rows].astype(np.float64)
    baseline_error = float(np.mean((reference-truth)**2))
    rng = np.random.default_rng(seed)
    gains, changes = [], []
    for _ in range(maps):
        alternatives = np.asarray([rng.choice(eligible[int(index)]) for index in rows])
        swapped = condition[rows].copy()
        swapped[:, :2048] = condition[alternatives, :2048]
        replaced = predict_raw(model, tensors, pre[rows], swapped).astype(np.float64)
        gains.append(float(np.mean((replaced-truth)**2))-baseline_error)
        changes.append(float(np.sqrt(np.mean((replaced-reference)**2))))
    result["compound_mse_increase"] = float(np.mean(gains))
    result["prediction_rms_change"] = float(np.mean(changes))
    return result


def analyze_file(source, checkpoint, normalizer, data, output, *, maps=32, seed=41):
    model, tensors = load_predictor(source, checkpoint, normalizer)
    result = compound_analysis(model, tensors, load_arrays(data, target=True), maps=maps, seed=seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result
