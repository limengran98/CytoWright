"""Run a CPU walkthrough from response feedback to a saved predictor."""
from pathlib import Path
import argparse
import json
import shutil
import types

import numpy as np
import torch

from .feedback.core import metrics, predict_raw
from .feedback.editing import prepare, refine
from .inference import compound_analysis, load_predictor, save_normalizer
from .paths import project_path


def run_demo(output, *, seed=41):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    context_dim, target_dim = 591, 1568
    coefficients = rng.normal(0, 0.4, size=(8, target_dim))
    control_effect = rng.normal(0, 0.02, size=(4, target_dim))

    def samples(count):
        contexts = np.repeat(rng.normal(size=(count//8, context_dim)), 8, axis=0)
        chemical = np.zeros((count, 2048))
        chemical[:, :8] = rng.normal(size=(count, 8))
        dose = np.repeat(rng.normal(size=(count//8, 1)), 8, axis=0)
        response = chemical[:, :8]@coefficients + contexts[:, :4]@control_effect + rng.normal(0, 0.02, size=(count, target_dim))
        return {"pre": contexts.astype(np.float32), "condition": np.column_stack([chemical, dose]).astype(np.float32),
                "target": response.astype(np.float32)}

    training, validation, query = samples(96), samples(48), samples(64)
    tensors = prepare(training, validation)
    source = project_path("programs/demo/predictor.py")
    module = types.ModuleType("cytowright_demo_predictor")
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    parent = module.build_model({"context_dim": context_dim, "chemical_dim": 2048, "dose_dim": 1,
                                "target_dim": target_dim, "cp_dim": context_dim, "l1000_dim": 977}).eval()
    before = predict_raw(parent, tensors, query["pre"], query["condition"])
    corrected, record = refine(parent, tensors, validation, seed)
    prediction = predict_raw(corrected, tensors, query["pre"], query["condition"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output/"predictor.py")
    torch.save({"model_state_dict": corrected.state_dict(), "context": record["context"],
                "standardization_fingerprint": tensors.standardizer.fingerprint}, output/"model.pt")
    save_normalizer(tensors.standardizer, output/"normalizer.npz")
    np.savez_compressed(output/"query.npz", **query)
    np.save(output/"predictions.npy", prediction)
    restored, restored_tensors = load_predictor(output/"predictor.py", output/"model.pt", output/"normalizer.npz")
    replay = predict_raw(restored, restored_tensors, query["pre"], query["condition"])
    summary = {"status": "complete", "example": "synthetic cellular-response feedback", "seed": seed,
               "training_samples": 96, "validation_samples": 48, "query_samples": 64,
               "before": metrics(before, query["target"], cp_dim=context_dim),
               "after": metrics(prediction, query["target"], cp_dim=context_dim),
               "selected_candidate": record["selected_candidate"],
               "native_reload_max_error": float(np.max(np.abs(replay-prediction))),
               "compound_analysis": compound_analysis(restored, restored_tensors, query, maps=32, seed=seed)}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args(argv)
    result = run_demo(args.output, seed=args.seed)
    print("status: "+result["status"])
    print(f"original_mse: {result['before']['mse']:.4f}")
    print(f"corrected_mse: {result['after']['mse']:.4f}")
    print(f"corrected_pcc: {result['after']['global_pcc']:.4f}")
    print(f"native_reload_max_error: {result['native_reload_max_error']:.1f}")


if __name__ == "__main__":
    main()
