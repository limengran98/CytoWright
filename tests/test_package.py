import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cytowright.feedback.linear import solve_ridge
from cytowright.inference import load_arrays, load_predictor, replacement_groups
from cytowright.paths import project_path


class CoreChecks(unittest.TestCase):
    def test_resources_and_initial_candidate(self):
        from cytowright.joint_response_runtime import load_runtime_config, load_starting_candidate
        from cytowright.tasks import get_task
        config = load_runtime_config("configs/joint_response_runtime.json")
        self.assertTrue(load_starting_candidate("h0", config).candidate_hash)
        for task in ["cpg036_cp_plate_control_context", "cpg047_cp_plate_control_context"]:
            self.assertEqual(get_task(task)["adapter"], "cpg_replicate_profiles")
        self.assertTrue(project_path("programs/bbbc036/predictor.py").is_file())
        self.assertTrue(project_path("programs/bbbc047/predictor.py").is_file())

    def test_ridge_normal_equations(self):
        rng = np.random.default_rng(17)
        for samples, features in [(7, 4), (4, 7)]:
            x, y = rng.normal(size=(samples, features)), rng.normal(size=(samples, 3))
            alpha = 0.7*np.sum(x*x)/features + 1e-8
            expected = np.linalg.solve(x.T@x+alpha*np.eye(features), x.T@y)
            np.testing.assert_allclose(solve_ridge(x, y, 0.7), expected, rtol=1e-12, atol=1e-12)

    def test_affine_prediction_interfaces(self):
        from cytowright.feedback.core import interface_check
        self.assertEqual(interface_check()["status"], "pass")

    def test_replacement_matches_context_dose_and_distinct_compounds(self):
        pre = np.array([[1, 2], [1, 2], [1, 2], [3, 4]], dtype=np.float32)
        condition = np.zeros((4, 2049), dtype=np.float32)
        condition[:, 0], condition[:, -1] = [1, 2, 3, 4], [1, 1, 2, 1]
        self.assertEqual(replacement_groups(pre, condition), {0: [1], 1: [0]})

    def test_named_data_exports_preserve_rows(self):
        from cytowright.workflow import export_arrays
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pre = np.arange(15, dtype=np.float32).reshape(5, 3)
            cache = root/"cache.npz"
            np.savez(cache, pre=pre, condition=np.zeros((5, 2049), dtype=np.float32),
                     target=np.ones((5, 7), dtype=np.float32), fold=np.arange(5))
            with patch("cytowright.workflow._manifest_verified_cache", return_value=(cache, {}, {}, "")):
                result = export_arrays("bbbc036", root/"arrays")
            self.assertEqual(result["samples"], {"training": 2, "validation": 1, "main_test": 1, "additional_test": 1})
            with np.load(root/"arrays/training.npz") as arrays:
                np.testing.assert_array_equal(arrays["pre"], pre[:2])


class DemoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cytowright.demo import run_demo
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.result = run_demo(cls.root, seed=41)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_correction_improves_query_prediction(self):
        self.assertLess(self.result["after"]["mse"], self.result["before"]["mse"])
        self.assertGreater(self.result["after"]["global_pcc"], 0.9)

    def test_native_reload_and_frozen_control_encoder(self):
        self.assertEqual(self.result["native_reload_max_error"], 0.0)
        import types
        module = types.ModuleType("demo_source")
        exec((self.root/"predictor.py").read_text(), module.__dict__)
        torch.manual_seed(41)
        parent = module.build_model({"context_dim": 591, "target_dim": 1568})
        state = torch.load(self.root/"model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        for name, value in parent.control_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, state["base.control_encoder."+name]))

    def test_compound_contribution_on_matched_query_samples(self):
        result = self.result["compound_analysis"]
        self.assertEqual(result["matched_samples"], 64)
        self.assertGreater(result["compound_mse_increase"], 0)
        self.assertGreater(result["prediction_rms_change"], 0)

    def test_prediction_command_reloads_complete_model(self):
        from cytowright.cli import main
        output = self.root/"reloaded.npy"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["predict", "--source", str(self.root/"predictor.py"), "--checkpoint", str(self.root/"model.pt"),
                  "--normalizer", str(self.root/"normalizer.npz"), "--data", str(self.root/"query.npz"),
                  "--output", str(output)])
        np.testing.assert_array_equal(np.load(output), np.load(self.root/"predictions.npy"))


class TrainingChecks(unittest.TestCase):
    def test_reference_program_training_and_inference(self):
        from cytowright.joint_response_runtime import CPGPartition, CPGFold13Arrays
        from cytowright.tasks import get_task
        from cytowright.workflow import train_program
        torch.set_num_threads(2)
        rng = np.random.default_rng(37)
        def part(n, roles):
            pre = rng.normal(size=(n, 591)).astype(np.float32)
            condition = rng.normal(size=(n, 2049)).astype(np.float32)
            target = rng.normal(size=(n, 1568)).astype(np.float32)
            return CPGPartition(pre, condition, target, target[:, :591], target[:, 591:],
                                tuple("row"+str(i) for i in range(n)), np.asarray(roles))
        training = part(16, [0]*8+[1]*8)
        validation = part(8, [2]*8)
        task_id = "cpg036_cp_plate_control_context"
        arrays = CPGFold13Arrays(task_id, training, validation, get_task(task_id)["condition_layout"], "fixture", {})
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"model"
            with patch("cytowright.workflow.load_fold13_arrays", return_value=arrays):
                result = train_program("bbbc036", project_path("programs/bbbc036/predictor.py"),
                                       output, seed=41, device="cpu", epochs=1)
            self.assertEqual(result["status"], "trained")
            model, tensors = load_predictor(output/"predictor.py", output/"model.pt", output/"normalizer.npz")
            from cytowright.feedback.core import predict_raw
            prediction = predict_raw(model, tensors, validation.pre, validation.condition)
            self.assertEqual(prediction.shape, (8, 1568))
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
