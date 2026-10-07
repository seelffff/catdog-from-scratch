"""Freeze a metadata-only V2 split from the original supplied 9,972 images.

V1 validation is preserved. V2 test is sampled by duplicate groups from V1
train. V1 test becomes V2 train. This script never opens image files, imports
image/ML libraries, trains a model, or evaluates predictions.

IMPORTANT: every V2 network must start with newly randomized parameters.
V1 weights trained on the images that become V2 test, so they are forbidden.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"cat": "0", "dog": "1"}
SPLITS = ("train", "val", "test")
EXCLUDED_DEMONSTRATION_IMAGES = (
    "data/raw/train/cat.0.jpg",
    "data/raw/train/dog.1.jpg",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_rows(rows: list[dict]) -> dict:
    by_path, groups, hashes = {}, defaultdict(list), defaultdict(list)
    for row in rows:
        path = PurePosixPath(row["relative_path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest contains an unsafe image path")
        if row["relative_path"] in by_path:
            raise ValueError("Manifest contains repeated image paths")
        by_path[row["relative_path"]] = row
        if row["split"] not in SPLITS or LABELS.get(row["class_name"]) != row["label"]:
            raise ValueError("Manifest contains an unexpected split or inconsistent class label")
        if not path.name.startswith(row["class_name"] + "."):
            raise ValueError("Filename and class label disagree")
        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise ValueError("Manifest has nonpositive image dimensions")
        if len(row["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in row["sha256"]):
            raise ValueError("Manifest contains an invalid recorded image SHA-256")
        groups[row["group_id"]].append(row)
        hashes[row["sha256"]].append(row)
    for members in groups.values():
        if len({row["split"] for row in members}) != 1:
            raise ValueError("A duplicate group crosses splits")
        if len({row["label"] for row in members}) != 1:
            raise ValueError("A duplicate group has conflicting labels")
    for members in hashes.values():
        if len({row["group_id"] for row in members}) != 1:
            raise ValueError("An exact-image hash appears in different duplicate groups")
    return {"by_path": by_path, "groups": dict(groups)}


def stratified_targets(class_counts: dict[str, int], requested: int) -> dict[str, int]:
    total = sum(class_counts.values())
    if not 0 < requested < total:
        raise ValueError("V2 test size must be positive and smaller than the original train split")
    targets = {name: requested * count // total for name, count in class_counts.items()}
    remainder = requested - sum(targets.values())
    # Largest-remainder apportionment, with deterministic class-name tie breaking.
    ordered = sorted(class_counts, key=lambda name: (-(requested * class_counts[name] % total), name))
    for name in ordered[:remainder]:
        targets[name] += 1
    return targets


def choose_whole_groups(class_groups: list[tuple[str, int]], target: int, generator: random.Random) -> set[str]:
    """Find an exact-size whole-group subset in a fixed seeded random ordering."""
    ordered = sorted(class_groups)
    generator.shuffle(ordered)
    reachable = [False] * (target + 1)
    parents = [None] * (target + 1)
    reachable[0] = True
    for group_id, size in ordered:
        for subtotal in range(target - size, -1, -1):
            candidate = subtotal + size
            if reachable[subtotal] and not reachable[candidate]:
                reachable[candidate] = True
                parents[candidate] = (subtotal, group_id)
        if reachable[target]:
            break
    if not reachable[target]:
        raise ValueError(f"Duplicate groups cannot produce exactly {target} images for one class")
    selected = set()
    subtotal = target
    while subtotal:
        subtotal, group_id = parents[subtotal]
        selected.add(group_id)
    return selected


def counts(rows: list[dict]) -> dict:
    result = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        by_class = Counter(row["class_name"] for row in selected)
        result[split] = {
            "images": len(selected), "cat": by_class["cat"], "dog": by_class["dog"],
            "unique_groups": len({row["group_id"] for row in selected}),
        }
    return result


def publish_exclusively(files: list[tuple[Path, bytes]]) -> None:
    temporary_paths, created_paths = [], []
    try:
        for path, payload in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix="v2-split-", delete=False) as file:
                file.write(payload)
                temporary = Path(file.name)
            temporary_paths.append((temporary, path))
        for temporary, path in temporary_paths:
            os.link(temporary, path)  # Atomic, and fails rather than overwrite an existing file.
            created_paths.append(path)
    except BaseException:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)


def prepare_v2_split(source: Path, output: Path, report: Path, seed: int = 9001, test_size: int = 1000,
                     excluded_demonstration_images: tuple[str, ...] = EXCLUDED_DEMONSTRATION_IMAGES) -> dict:
    source, output, report = source.resolve(), output.resolve(), report.resolve()
    if len({source, output, report}) != 3:
        raise ValueError("Source, new manifest and report must use different paths")
    if output.parent.parent != source.parent.parent:
        raise ValueError("New manifest must remain directly inside data/ so ImageDataset.root stays the project directory")
    if output.exists() or report.exists():
        raise FileExistsError("V2 split is frozen: use fresh output/report paths; existing artifacts will not be overwritten")
    source_payload = source.read_bytes()
    source_hash = digest(source_payload)
    reader = csv.DictReader(io.StringIO(source_payload.decode("utf-8"), newline=""))
    fieldnames, rows = reader.fieldnames, list(reader)
    if not rows or not fieldnames:
        raise ValueError("V1 manifest is empty")
    original = check_rows(rows)
    excluded_groups = set()
    for path in excluded_demonstration_images:
        if path not in original["by_path"] or original["by_path"][path]["split"] != "train":
            raise ValueError("Each excluded demonstration image must belong to V1 train")
        excluded_groups.add(original["by_path"][path]["group_id"])
    class_groups = defaultdict(list)
    for group_id, members in original["groups"].items():
        if members[0]["split"] == "train" and group_id not in excluded_groups:
            class_groups[members[0]["class_name"]].append((group_id, len(members)))
    class_counts = {name: sum(size for _, size in class_groups[name]) for name in LABELS}
    targets = stratified_targets(class_counts, test_size)
    generator = random.Random(seed)
    new_test_groups = set()
    for name in sorted(LABELS):
        new_test_groups.update(choose_whole_groups(class_groups[name], targets[name], generator))
    new_rows = []
    for old in rows:
        row = dict(old)
        if old["split"] == "test":
            row["split"] = "train"
        elif old["split"] == "train" and old["group_id"] in new_test_groups:
            row["split"] = "test"
        new_rows.append(row)
    prepared = check_rows(new_rows)
    if set(original["by_path"]) != set(prepared["by_path"]):
        raise AssertionError("V2 must use exactly the original supplied image paths")
    old_val = [row for row in rows if row["split"] == "val"]
    new_val = [row for row in new_rows if row["split"] == "val"]
    if old_val != new_val:
        raise AssertionError("V1 validation rows must remain identical")
    for path, old in original["by_path"].items():
        new = prepared["by_path"][path]
        if {key: value for key, value in old.items() if key != "split"} != {
            key: value for key, value in new.items() if key != "split"
        }:
            raise AssertionError("Only split assignments may change")
        if old["split"] == "test" and new["split"] != "train":
            raise AssertionError("Every old test image must become V2 train")
        if new["split"] == "test" and old["split"] != "train":
            raise AssertionError("Every new test image must come from original train")
        if old["group_id"] in excluded_groups and new["split"] != "train":
            raise AssertionError("Demonstration images and their whole duplicate groups must remain train")
    new_counts = counts(new_rows)
    if new_counts["test"]["images"] != test_size:
        raise AssertionError("V2 test size differs from frozen request")
    for name, target in targets.items():
        if new_counts["test"][name] != target:
            raise AssertionError("V2 test class counts differ from stratified targets")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)
    manifest_payload = buffer.getvalue().encode("utf-8")
    transitions = Counter((old["split"], new["split"]) for old, new in zip(rows, new_rows, strict=True))
    plan = {
        "revision": "V2", "seed": seed, "requested_test_images": test_size,
        "dataset_policy": "Only the original supplied 9972-image dataset; no new or external images",
        "source_manifest": str(source), "source_manifest_sha256": source_hash,
        "output_manifest": str(output), "output_manifest_sha256": digest(manifest_payload),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "source_counts": counts(rows), "v2_counts": new_counts,
        "total_images": len(new_rows), "class_to_label": {"cat": 0, "dog": 1},
        "test_stratification_source": "Eligible V1 train after demonstration-group exclusions; largest-remainder integer apportionment",
        "eligible_v1_train_class_counts": class_counts,
        "test_class_targets": targets,
        "excluded_demonstration_images": list(excluded_demonstration_images),
        "excluded_demonstration_group_ids": sorted(excluded_groups),
        "excluded_demonstration_group_member_paths": [
            row["relative_path"] for row in rows if row["group_id"] in excluded_groups
        ],
        "group_sampling": "Sorted group IDs per class; random.Random(seed) shuffle; exact-size subset sum in that fixed order",
        "group_policy": "All V1 exact/verified-near duplicate group_id members remain together",
        "transitions": {f"{old}_to_{new}": count for (old, new), count in sorted(transitions.items())},
        "old_validation_preserved": True,
        "old_test_policy": "All V1 test images are now V2 train; V1 metrics remain historical V1 results and are not V2 metrics",
        "new_test_policy": "V2 test is a new closed holdout from V1 train; no pixels or predictions accessed during preparation",
        "training_policy": {
            "required_initialization": "random",
            "v1_checkpoints_allowed": False,
            "reason": "V1 networks already trained on the former V1 train images that now form V2 test",
            "must_reinitialize_all_weights_and_buffers": True,
            "checkpoint_manifest_hash_must_match": digest(manifest_payload),
            "validation_only_selection": True,
            "final_test_policy": "Freeze all model/preprocessing/calibration choices before one final V2 test; never tune from that result",
        },
        "path_compatibility": {
            "image_dataset_root": str(output.parent.parent),
            "relative_paths_unchanged": True,
            "reason": "ImageDataset computes root as manifest.parent.parent; a nested data/v2/splits.csv would incorrectly resolve existing data/raw/train paths",
        },
        "checks": {
            "validation_rows_identical": True, "paths_labels_hashes_dimensions_preserved": True,
            "duplicate_groups_crossing_splits": 0, "exact_image_hashes_crossing_splits": 0,
            "path_sets_disjoint_between_splits": True, "group_sets_disjoint_between_splits": True,
            "only_original_image_paths_used": True, "all_old_test_now_train": True,
            "all_new_test_from_old_train": True, "image_files_opened": 0,
            "image_hashes_reused_from_v1_metadata": True,
            "demonstration_images_and_their_groups_remain_train": True,
        },
    }
    if digest(source.read_bytes()) != source_hash:
        raise RuntimeError("V1 manifest changed during preparation")
    publish_exclusively([(output, manifest_payload), (report, (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))])
    if digest(source.read_bytes()) != source_hash or digest(output.read_bytes()) != plan["output_manifest_sha256"]:
        raise RuntimeError("Published manifest hash verification failed")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/splits_v1.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/splits_v2.csv")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/v2/split_plan.json")
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--test-size", type=int, default=1000)
    args = parser.parse_args()
    plan = prepare_v2_split(args.source, args.output, args.report, args.seed, args.test_size)
    print(json.dumps({key: plan[key] for key in ("output_manifest", "output_manifest_sha256", "v2_counts", "transitions", "training_policy")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
