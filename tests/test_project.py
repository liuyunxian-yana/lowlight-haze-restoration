"""独立项目的物理模型、接口、数据和评估检查。"""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from dip_restore.core import linear_to_srgb, srgb_to_linear, validate
from dip_restore.datasets import Case, load, verify, write_rgb
from dip_restore.enhancement import ENHANCERS
from dip_restore.dehazing import DEHAZERS
from dip_restore.joint_model import JOINT_MODEL, inverse_scene, recover
from dip_restore.metrics import measure
from dip_restore.registry import ALL, parse, restore
from dip_restore.benchmark import run


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(11).integers(15, 190, (24, 28, 3), dtype=np.uint8)

    def test_all_registered_methods_keep_contract(self):
        original = self.image.copy()
        for name in ALL:
            with self.subTest(name=name):
                validate(restore(self.image, name), self.image.shape)
        np.testing.assert_array_equal(original, self.image)
        self.assertEqual((len(ENHANCERS), len(DEHAZERS)), (7, 3))
        self.assertIn("model_full", JOINT_MODEL)

    def test_order_and_color_conversion(self):
        self.assertEqual(parse("gamma>dcp"), ("gamma", "dcp"))
        self.assertFalse(np.array_equal(restore(self.image, "gamma>dcp"), restore(self.image, "dcp>gamma")))
        with self.assertRaises(ValueError): parse("gamma>side2020")
        linear = srgb_to_linear(self.image.astype(np.float32) / 255)
        np.testing.assert_allclose(linear_to_srgb(linear), self.image / 255, atol=2e-6)

    def test_inverse_matches_forward_model(self):
        rng = np.random.default_rng(4)
        scene = rng.uniform(.1, .8, (16, 17, 3)).astype(np.float32)
        e = np.full(scene.shape[:2], .4, np.float32)
        t = np.full(scene.shape[:2], .65, np.float32)
        a = np.array([.72, .75, .78], np.float32)
        q = e * (1 - t)
        y = e[..., None] * (t[..., None] * scene + (1-t[..., None])*a)
        recovered, _ = inverse_scene(y, e, q, a, noise_sigma=0)
        np.testing.assert_allclose(recovered, scene, atol=2e-6)

    def test_joint_ablation_bounds(self):
        for mode in ("full", "global_e", "light_only", "haze_only", "unregularized"):
            with self.subTest(mode=mode):
                output, fields = recover(self.image, mode=mode, return_fields=True)
                validate(output, self.image.shape)
                self.assertTrue(np.all(fields["exposure"] >= .025))
                self.assertTrue(np.all(fields["veil_coefficient"] <= .85*fields["exposure"]+1e-6))
                self.assertTrue(np.all(fields["transmission"] >= .15-1e-6))

    def test_data_and_metrics(self):
        self.assertEqual(len(load("synthetic", "val", "all")), 54)
        self.assertEqual(len(load("synthetic", "test", "all")), 108)
        self.assertEqual(len(load("synthetic", "test", "joint")), 36)
        self.assertEqual(len(load("nthazy")), 25)
        self.assertEqual(len(load("gta5")), 77)
        case = load("synthetic", "test", "joint")[0]
        x, y = verify(case)
        self.assertEqual(x.shape, y.shape)
        with self.assertRaises(ValueError):
            verify(Case(case.id,case.source_id,case.split,case.kind,case.severity,case.input_path,case.reference_path,"bad",case.reference_sha256))
        same = measure(self.image,self.image)
        self.assertTrue(np.isinf(same["psnr_db"]))
        self.assertAlmostEqual(same["ssim_rgb"],1,places=8)

    def test_unpaired_batch_does_not_invent_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); write_rgb(root/"input/sample.png",self.image)
            self.assertEqual(run(dataset="external",methods=("model_global_e",),output=root/"out",input_dir=root/"input",repeats=1,compare_count=0),0)
            with (root/"out/per_image.csv").open(encoding="utf-8-sig") as file:
                row=next(csv.DictReader(file))
            self.assertEqual(row["psnr_db"],"")
            self.assertEqual(row["ssim_rgb"],"")


if __name__ == "__main__": unittest.main()
