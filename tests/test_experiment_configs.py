"""Regression tests for experiment configurations and entry points."""

import contextlib
import io
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    ("ex_darcyrough", "ex_darcyrough.py", "train_model", 512, 256, 256,
     "darcyrough_res256.pt"),
    ("ex_darcysmooth", "ex_darcysmooth.py", "train_model", 421, 208, 211,
     "darcysmooth_res211.pt"),
    ("ex_multiscale", "ex_multiscale.py", "train_model", 1023, 256, 256,
     "multiscale_res256.pt"),
    ("ex_ns", "ex_ns.py", "train_NS_model", 64, 64, 64,
     "NS_1e-5_N1200.pt"),
    ("ex_fno_multiscale", "FNO_multiscale.py", "train_model", 256, None, 256,
     "FNO_multiscale_res256.pt"),
)


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.trainer = types.ModuleType("hano.trainer")
        self.trainer.train_model = Mock()
        self.trainer.train_NS_model = Mock()
        self.module_patch = patch.dict(sys.modules, {"hano.trainer": self.trainer})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))

    def test_configurations_and_entry_points(self):
        for module, wrapper, trainer_name, res_input, res_att, res_output, model_name in EXPERIMENTS:
            with self.subTest(experiment=module):
                path = ROOT / "experiments" / (module + ".py")
                namespace = runpy.run_path(str(path))
                config = namespace["build_config"]()
                self.assertEqual(config["res_input"], res_input)
                self.assertEqual(config.get("res_att"), res_att)
                self.assertEqual(config["res_output"], res_output)
                self.assertEqual(config["model_name"], model_name)
                self.assertEqual(config["result_name"], model_name[:-3] + ".pkl")
                self.assertEqual(config["mat_name"], model_name[:-3] + ".mat")
                self.assertEqual(Path(config["Data_path"]).resolve(), ROOT / "data")
                self.assertEqual(Path(config["model_save_path"]).resolve(), ROOT / "models")

                config["train_len"] = -1
                self.assertNotEqual(namespace["build_config"]()["train_len"], -1)

                trainer = getattr(self.trainer, trainer_name)
                for entry_point in (path, ROOT / wrapper):
                    trainer.reset_mock()
                    with contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(entry_point), run_name="__main__")
                    trainer.assert_called_once_with(namespace["build_config"]())


if __name__ == "__main__":
    unittest.main()
