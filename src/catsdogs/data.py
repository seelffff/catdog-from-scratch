"""Validate image data and create reproducible, duplicate-aware splits.

No images or labels are modified. Exact and conservatively verified near
duplicates stay in the same split; conflicting-label groups are quarantined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASS_TO_LABEL = {"cat": 0, "dog": 1}
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = (
    "relative_path", "label", "class_name", "split", "group_id",
    "width", "height", "sha256", "phash",
)

# Only the low-frequency rows of an orthonormal 32-point DCT are needed.
_positions = np.arange(32, dtype=np.float32)
_frequencies = np.arange(8, dtype=np.float32)[:, None]
_DCT = np.cos(np.pi * (_positions + 0.5) * _frequencies / 32)
_DCT *= np.sqrt(2 / 32)
_DCT[0] /= np.sqrt(2)


def _inspect_image(path: Path, project_root: Path) -> dict:
    relative_path = path.relative_to(project_root).as_posix()
    class_name = path.name.split(".", 1)[0]
    if class_name not in CLASS_TO_LABEL:
        return {"relative_path": relative_path, "error": "Unknown filename label"}
    try:
        with Image.open(path) as source:
            # Fully decode: a successful header read alone does not validate JPEGs.
            source.load()
            rgb = ImageOps.exif_transpose(source).convert("RGB")
        width, height = rgb.size
        pixels = np.asarray(rgb, dtype=np.uint8)
        digest = hashlib.sha256()
        digest.update(f"{width}x{height}:RGB:".encode())
        digest.update(pixels.tobytes())
        gray = np.asarray(
            rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        coefficients = _DCT @ gray @ _DCT.T
        bits = (coefficients.ravel() > np.median(coefficients.ravel()[1:]))
        bits[0] = False  # Ignore mean brightness (the DC coefficient).
        perceptual_hash = sum(int(bit) << i for i, bit in enumerate(bits))
        thumbnail = np.asarray(
            rgb.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.uint8,
        )
        return {
            "relative_path": relative_path,
            "label": CLASS_TO_LABEL[class_name],
            "class_name": class_name,
            "width": width,
            "height": height,
            "sha256": digest.hexdigest(),
            "phash": f"{perceptual_hash:016x}",
            "_phash_int": perceptual_hash,
            "_thumbnail": thumbnail,
        }
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return {"relative_path": relative_path, "error": f"{type(error).__name__}: {error}"}


class _Groups:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def _find_duplicate_groups(records: list[dict]) -> tuple[_Groups, list[list[str]], list[dict]]:
    groups = _Groups(len(records))
    exact = defaultdict(list)
    for index, record in enumerate(records):
        exact[record["sha256"]].append(index)
    exact_groups = []
    for indices in exact.values():
        if len(indices) > 1:
            exact_groups.append([records[i]["relative_path"] for i in indices])
            for other in indices[1:]:
                groups.join(indices[0], other)

    # Five disjoint hash bands guarantee a matching band for Hamming distance <= 4.
    # Hash closeness alone is not enough: RGB thumbnails and aspect ratio also
    # must agree. This intentionally favors precision over finding every crop.
    buckets = defaultdict(list)
    near_pairs = []
    for index, record in enumerate(records):
        value = record["_phash_int"]
        candidates = set()
        keys = []
        for band, (offset, size) in enumerate(((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))):
            key = (band, (value >> offset) & ((1 << size) - 1))
            candidates.update(buckets[key])
            keys.append(key)
        for other_index in sorted(candidates):
            other = records[other_index]
            if groups.find(index) == groups.find(other_index):
                continue
            distance = (value ^ other["_phash_int"]).bit_count()
            if distance > 4:
                continue
            ratio = (record["width"] / record["height"]) / (other["width"] / other["height"])
            if not 0.95 <= ratio <= 1.05:
                continue
            first = record["_thumbnail"].astype(np.float32).ravel()
            second = other["_thumbnail"].astype(np.float32).ravel()
            mean_absolute_difference = float(np.mean(np.abs(first - second)))
            if mean_absolute_difference > 6.0:
                continue
            first_centered, second_centered = first - first.mean(), second - second.mean()
            denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
            correlation = float(first_centered @ second_centered / denominator) if denominator else 0.0
            if correlation < 0.985:
                continue
            groups.join(index, other_index)
            near_pairs.append({
                "first": record["relative_path"],
                "second": other["relative_path"],
                "phash_distance": distance,
                "thumbnail_mae_0_255": round(mean_absolute_difference, 4),
                "thumbnail_correlation": round(correlation, 6),
            })
        for key in keys:
            buckets[key].append(index)
    return groups, exact_groups, near_pairs


def _assign_splits(records: list[dict], groups: _Groups, seed: int) -> tuple[list[dict], list[dict]]:
    members = defaultdict(list)
    for index in range(len(records)):
        members[groups.find(index)].append(index)
    quarantined = []
    groups_by_class = defaultdict(list)
    for root, indices in sorted(members.items()):
        labels = {records[i]["label"] for i in indices}
        if len(labels) > 1:
            quarantined.append({
                "reason": "Duplicate images have conflicting filename labels; manual review required",
                "files": [records[i]["relative_path"] for i in indices],
            })
            continue
        groups_by_class[next(iter(labels))].append((root, indices))
    random_generator = random.Random(seed)
    assignments = {}
    for label in sorted(groups_by_class):
        class_groups = groups_by_class[label]
        total = sum(len(indices) for _, indices in class_groups)
        val_count, test_count = round(total * 0.1), round(total * 0.1)
        targets = {"train": total - val_count - test_count, "val": val_count, "test": test_count}
        counts = dict.fromkeys(SPLITS, 0)
        random_generator.shuffle(class_groups)
        # Place larger duplicate groups first; shuffled ordering breaks equal-size ties.
        class_groups.sort(key=lambda item: len(item[1]), reverse=True)
        for root, indices in class_groups:
            size = len(indices)
            possible = [split for split in SPLITS if counts[split] + size <= targets[split]]
            if not possible:
                possible = list(SPLITS)
            split = max(possible, key=lambda name: (targets[name] - counts[name]) / max(targets[name], 1))
            counts[split] += size
            for index in indices:
                assignments[index] = (split, f"group_{root:05d}")
    manifest = []
    for index, record in enumerate(records):
        if index not in assignments:
            continue
        split, group_id = assignments[index]
        row = {key: record[key] for key in MANIFEST_COLUMNS if key in record}
        row.update(split=split, group_id=group_id)
        manifest.append(row)
    return manifest, quarantined


def read_manifest(path: Path | str, split: str | None = None) -> list[dict]:
    """Read manifest rows, converting label and image dimensions to integers."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        rows = [row for row in rows if row["split"] == split]
    for row in rows:
        for field in ("label", "width", "height"):
            row[field] = int(row[field])
    return rows


def audit_dataset(
    data_dir: Path | str,
    manifest_path: Path | str,
    report_path: Path | str,
    project_root: Path | str | None = None,
    seed: int = 42,
    workers: int = 8,
) -> dict:
    """Decode every image, group duplicates, and save a fixed stratified split."""
    data_dir, manifest_path, report_path = map(Path, (data_dir, manifest_path, report_path))
    data_dir = data_dir.resolve()
    project_root = Path(project_root).resolve() if project_root else data_dir.parent.parent.parent
    paths = sorted(path for path in data_dir.iterdir() if path.is_file())
    if not paths:
        raise ValueError(f"No files found in {data_dir}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspections = list(executor.map(lambda path: _inspect_image(path, project_root), paths))
    invalid = [item for item in inspections if "error" in item]
    valid = [item for item in inspections if "error" not in item]
    if not valid:
        raise ValueError("No valid labeled images found")
    groups, exact_groups, near_pairs = _find_duplicate_groups(valid)
    manifest, quarantined = _assign_splits(valid, groups, seed)
    group_splits = defaultdict(set)
    for row in manifest:
        group_splits[row["group_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values()), "Duplicate group leakage"
    assert len({row["relative_path"] for row in manifest}) == len(manifest), "Repeated manifest paths"

    dimensions = np.array([(item["width"], item["height"]) for item in valid])
    split_counts = {
        split: {
            "total": sum(row["split"] == split for row in manifest),
            **{name: sum(row["split"] == split and row["class_name"] == name for row in manifest)
               for name in CLASS_TO_LABEL},
        }
        for split in SPLITS
    }
    report = {
        "dataset_directory": data_dir.relative_to(project_root).as_posix(),
        "seed": seed,
        "class_to_label": CLASS_TO_LABEL,
        "total_files": len(paths),
        "valid_images": len(valid),
        "invalid_images": invalid,
        "filename_label_counts": dict(Counter(record["class_name"] for record in valid)),
        "image_dimensions": {
            "width_min_median_max": [int(dimensions[:, 0].min()), float(np.median(dimensions[:, 0])), int(dimensions[:, 0].max())],
            "height_min_median_max": [int(dimensions[:, 1].min()), float(np.median(dimensions[:, 1])), int(dimensions[:, 1].max())],
        },
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_extra_images": sum(len(group) - 1 for group in exact_groups),
        "verified_near_duplicate_pairs": near_pairs,
        "duplicate_grouping_policy": {
            "exact": "SHA-256 of EXIF-oriented, decoded RGB pixels and dimensions",
            "near": "64-bit DCT perceptual hash Hamming distance <=4, aspect ratio within 5%, 32x32 RGB MAE <=6/255, RGB correlation >=0.985",
            "leakage_prevention": "All connected exact/verified-near duplicate members are assigned to one split",
            "limitations": "Conservative near matching may miss crops, mirrors or heavily edited duplicates; filename labels are not visually corrected",
        },
        "quarantined_conflicting_label_groups": quarantined,
        "manifest_images": len(manifest),
        "unique_duplicate_groups": len(group_splits),
        "split_counts": split_counts,
        "duplicate_groups_crossing_splits": 0,
        "test_policy": "Test split is reserved for final evaluation; model/threshold selection must use val only",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    from .paths import PROJECT_ROOT
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "data/raw/train")
    parser.add_argument("--manifest", type=Path, default=project_root / "data/splits.csv")
    parser.add_argument("--report", type=Path, default=project_root / "artifacts/data_audit.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    report = audit_dataset(arguments.data_dir, arguments.manifest, arguments.report, project_root, arguments.seed, arguments.workers)
    summary = {key: report[key] for key in ("total_files", "valid_images", "manifest_images", "split_counts", "exact_duplicate_extra_images", "duplicate_groups_crossing_splits")}
    summary["invalid_images"] = len(report["invalid_images"])
    summary["verified_near_duplicate_pairs"] = len(report["verified_near_duplicate_pairs"])
    summary["quarantined_conflicting_label_groups"] = len(report["quarantined_conflicting_label_groups"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
