"""Guard evaluation independence and audited-data behavior without training a model."""

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from catsdogs.data import audit_dataset, read_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_dataset(root, count_per_class=20):
    directory = root / "data/raw/train"
    directory.mkdir(parents=True)
    generator = np.random.default_rng(5)
    for class_name in ("cat", "dog"):
        for number in range(count_per_class):
            pixels = generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / f"{class_name}.{number}.jpg")
    return directory


def audit(root, directory):
    manifest = root / "data/splits.csv"
    report = root / "artifacts/data_audit.json"
    result = audit_dataset(directory, manifest, report, project_root=root, workers=2)
    return result, manifest, read_manifest(manifest)


def test_audit_rejects_corruption_quarantines_conflicts_and_keeps_exact_copies_together(tmp_path):
    directory = make_dataset(tmp_path)
    shutil.copyfile(directory / "cat.0.jpg", directory / "cat.90.jpg")
    shutil.copyfile(directory / "cat.1.jpg", directory / "dog.90.jpg")
    # This JPEG still has a readable header, but fails when decoded fully.
    valid_bytes = (directory / "cat.2.jpg").read_bytes()
    (directory / "cat.999.jpg").write_bytes(valid_bytes[:-200])
    with Image.open(directory / "dog.0.jpg") as source:
        source.save(directory / "unlabeled.jpg")
    result, manifest, rows = audit(tmp_path, directory)
    by_name = {Path(row["relative_path"]).name: row for row in rows}

    assert len(result["invalid_images"]) == 2
    invalid_names = {Path(row["relative_path"]).name for row in result["invalid_images"]}
    assert invalid_names == {"cat.999.jpg", "unlabeled.jpg"}
    assert len(result["quarantined_conflicting_label_groups"]) == 1
    assert "cat.1.jpg" not in by_name
    assert "dog.90.jpg" not in by_name
    assert by_name["cat.0.jpg"]["split"] == by_name["cat.90.jpg"]["split"]
    assert by_name["cat.0.jpg"]["group_id"] == by_name["cat.90.jpg"]["group_id"]
    assert len(rows) == 40
    assert all(isinstance(row["label"], int) for row in rows)
    assert all(isinstance(row["width"], int) for row in rows)

    # Rerunning the audit with the same seed must preserve every assignment.
    first_manifest = manifest.read_bytes()
    audit(tmp_path, directory)
    assert manifest.read_bytes() == first_manifest


def test_resized_and_recompressed_scene_is_grouped_before_splitting(tmp_path):
    directory = make_dataset(tmp_path)
    y, x = np.mgrid[:120, :160]
    pixels = np.stack([
        127 + 80 * np.sin(x / 17) * np.cos(y / 13),
        127 + 75 * np.cos(x / 21 + y / 18),
        127 + 65 * np.sin(x / 23 + y / 19),
    ], axis=2).astype(np.uint8)
    original = Image.fromarray(pixels)
    original.save(directory / "cat.100.jpg", quality=95)
    original.resize((80, 60), Image.Resampling.LANCZOS).save(directory / "cat.101.jpg", quality=85)
    result, _, rows = audit(tmp_path, directory)
    by_name = {Path(row["relative_path"]).name: row for row in rows}

    assert by_name["cat.100.jpg"]["sha256"] != by_name["cat.101.jpg"]["sha256"]
    assert by_name["cat.100.jpg"]["group_id"] == by_name["cat.101.jpg"]["group_id"]
    assert by_name["cat.100.jpg"]["split"] == by_name["cat.101.jpg"]["split"]
    detected_pairs = [
        {Path(pair["first"]).name, Path(pair["second"]).name}
        for pair in result["verified_near_duplicate_pairs"]
    ]
    assert {"cat.100.jpg", "cat.101.jpg"} in detected_pairs


def test_project_manifest_is_stratified_disjoint_and_duplicate_aware():
    manifest = PROJECT_ROOT / "data/splits_v2.csv"
    report_file = PROJECT_ROOT / "reports/v2/data_audit.json"
    assert manifest.exists() and report_file.exists(), "The published V2 audit and split must be present"
    rows = read_manifest(manifest)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    by_path = {row["relative_path"]: row for row in rows}
    assert len(by_path) == len(rows) == report["manifest_images"]
    assert set(row["split"] for row in rows) == {"train", "val", "test"}
    group_splits = defaultdict(set)
    sha_splits = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
        sha_splits[row["sha256"]].add(row["split"])
        assert row["label"] == {"cat": 0, "dog": 1}[row["class_name"]]
        assert Path(row["relative_path"]).name.startswith(row["class_name"] + ".")
        assert row["width"] > 0 and row["height"] > 0
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert all(len(splits) == 1 for splits in sha_splits.values())
    for pair in report["verified_near_duplicate_pairs"]:
        assert by_path[pair["first"]]["group_id"] == by_path[pair["second"]]["group_id"]
        assert by_path[pair["first"]]["split"] == by_path[pair["second"]]["split"]
    for split, expected in report["split_counts"].items():
        selected = [row for row in rows if row["split"] == split]
        counts = Counter(row["class_name"] for row in selected)
        assert len(selected) == expected["total"]
        assert counts["cat"] == expected["cat"]
        assert counts["dog"] == expected["dog"]
    # The holdout proportions are within one duplicate group's rounding tolerance.
    largest_group = max(Counter(row["group_id"] for row in rows).values())
    for class_name in ("cat", "dog"):
        total = sum(row["class_name"] == class_name for row in rows)
        for split in ("val", "test"):
            count = sum(row["class_name"] == class_name and row["split"] == split for row in rows)
            assert abs(count - total * 0.1) <= largest_group
