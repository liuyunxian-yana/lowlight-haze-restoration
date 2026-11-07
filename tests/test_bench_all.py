"""统一批量入口的数据发现、无参考评估和视觉输出检查。"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import bench_all
from dip_restore.datasets import write_rgb


class BenchAllTests(unittest.TestCase):
    def test_catalog_contains_independent_and_registered_methods(self):
        names = bench_all.methods_catalog()
        self.assertTrue({"ceef", "rdt_source", "rdt_target", "zhang2014",
                         "model_full", "model_fusion_anchor", "gamma>dcp"}.issubset(names))

    def test_paired_and_unpaired_are_kept_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.full((24, 25, 3), 90, np.uint8)
            write_rgb(root / "paired" / "degraded" / "one.png", x)
            write_rgb(root / "paired" / "GT" / "one.png", x)
            write_rgb(root / "unpaired" / "input" / "two.png", x)
            write_rgb(root / "new" / "three.png", x)
            datasets, issues = bench_all.discover_metrics(root)
            self.assertFalse(issues)
            self.assertEqual(set(datasets), {"paired", "unpaired"})
            out = root / "out"
            failures = bench_all.run_metrics(root, out, {"identity": lambda a: a.copy()}, 2)
            self.assertFalse(failures)
            with (out / "paired" / "per_image.csv").open(encoding="utf-8-sig") as file:
                paired = next(csv.DictReader(file))
            with (out / "unpaired" / "per_image.csv").open(encoding="utf-8-sig") as file:
                unpaired = next(csv.DictReader(file))
            self.assertEqual(paired["ssim_rgb"], "1.0")
            self.assertEqual(unpaired["psnr_db"], "")
            self.assertNotEqual(unpaired["luma_entropy"], "")
            visual = bench_all.run_visual(root, root / "visual",
                                          {"identity": lambda a: a.copy(),
                                           "ceef": bench_all.methods_catalog()["ceef"]}, 2)
            self.assertFalse(visual)
            self.assertTrue((root / "visual" / "identity" / "three.png").is_file())
            self.assertTrue((root / "visual" / "ceef" / "three.png").is_file())
            self.assertTrue((root / "visual" / "original" / "three.png").is_file())
            self.assertTrue((root / "visual" / "comparison" / "three.png").is_file())
            self.assertEqual(bench_all.read_rgb(root / "visual" / "comparison" / "three.png").shape,
                             (272, 1280, 3))

    def test_pair_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            x=np.full((24,25,3),90,np.uint8)
            write_rgb(root / "paired" / "degraded" / "one.png", x)
            write_rgb(root / "paired" / "GT" / "other.png", x)
            datasets, issues=bench_all.discover_metrics(root)
            self.assertNotIn("paired",datasets)
            self.assertEqual(len(issues),3)


if __name__ == "__main__":
    unittest.main()
