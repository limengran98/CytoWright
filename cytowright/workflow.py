"""Prepare task arrays and fit executable cellular-response programs."""
from dataclasses import replace
from pathlib import Path
import json
import shutil
import types

import numpy as np

from .discovery_candidate import validate_candidate_source
from .inference import save_normalizer
from .joint_response_runtime import (JointResponseSettings, _manifest_verified_cache,
    load_fold13_arrays, load_runtime_config, prepare_joint_tensors, train_joint_candidate)
from .open_discovery import CandidateState
from .paths import project_path
from .tasks import default_cache_path, get_task, prepare_cpg_replicate_profiles

TASKS = {"bbbc036": "cpg036_cp_plate_control_context", "bbbc047": "cpg047_cp_plate_control_context"}


def export_arrays(task, output):
    cache, _, _, _ = _manifest_verified_cache(TASKS[task])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = {"training": [0, 1], "validation": [2], "main_test": [3], "additional_test": [4]}
    counts = {}
    with np.load(cache, allow_pickle=False) as archive:
        roles = archive["fold"]
        for name, values in groups.items():
            mask = np.isin(roles, values)
            arrays = {key: archive[key][mask].astype(np.float32, copy=False) for key in ["pre", "condition", "target"]}
            np.savez_compressed(output/(name+".npz"), **arrays)
            counts[name] = int(mask.sum())
    return {"status": "prepared", "task": task, "samples": counts, "output": str(output)}


def prepare_data(task, output=None):
    task_id = TASKS[task]
    target = default_cache_path(task_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    prepare_cpg_replicate_profiles(get_task(task_id), target)
    destination = project_path(output or "data/arrays/"+task)
    return export_arrays(task, destination)


def train_program(task, source, output, *, seed=41, device="cuda:0", epochs=None):
    config = load_runtime_config("configs/joint_response_runtime.json")
    settings = replace(JointResponseSettings.from_config(config, device=device), seed=seed)
    if epochs is not None:
        settings = replace(settings, max_epochs=epochs)
    source = Path(source)
    text = source.read_text()
    validate_candidate_source(text)
    module = types.ModuleType("cytowright_training_program")
    exec(compile(text, str(source), "exec"), module.__dict__)
    controls = {"max_epochs": settings.max_epochs, "batch_size": settings.batch_size,
                "patience": settings.early_stopping_patience, "min_delta": settings.early_stopping_min_delta,
                "gradient_clip_norm": settings.gradient_clip_norm}
    candidate = CandidateState(candidate_source=text, candidate_metadata=module.candidate_metadata(),
                               training_config=controls, parent_candidate_hash=None)
    arrays = load_fold13_arrays(TASKS[task])
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = train_joint_candidate("predictor", arrays=arrays, config=config, settings=settings,
                                  checkpoint_path=output/"model.pt", candidate_override=candidate)
    if source.resolve() != (output/"predictor.py").resolve():
        shutil.copyfile(source, output/"predictor.py")
    save_normalizer(prepare_joint_tensors(arrays).standardizer, output/"normalizer.npz")
    (output/"metrics.json").write_text(json.dumps(record, indent=2)+"\n")
    return {"status": "trained", "task": task, "validation": record["selection_metrics"],
            "epochs": record["completed_epochs"], "output": str(output)}
