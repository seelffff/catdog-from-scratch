"""Check train-only weighted sampling on synthetic manifest rows, without images."""

import csv
import hashlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from catsdogs.train import load_sampling_weights


@pytest.fixture
def rows():
    return [
        {"relative_path": "data/cat.0.jpg", "label": 0, "split": "train", "group_id": "cat-duplicate"},
        {"relative_path": "data/cat.1.jpg", "label": 0, "split": "train", "group_id": "cat-duplicate"},
        {"relative_path": "data/cat.2.jpg", "label": 0, "split": "train", "group_id": "cat-single"},
        {"relative_path": "data/dog.0.jpg", "label": 1, "split": "train", "group_id": "dog-first"},
        {"relative_path": "data/dog.1.jpg", "label": 1, "split": "train", "group_id": "dog-second"},
    ]


def write_weights(path, rows, values=None):
    values = values if values is not None else [1.0] * len(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "weight"])
        writer.writeheader()
        for row, weight in zip(rows, values, strict=True):
            writer.writerow({"relative_path": row["relative_path"], "weight": weight})
    return path


def test_sampling_balances_class_mass_and_duplicate_group_mass(rows, tmp_path):
    path = write_weights(tmp_path / "weights.csv", rows, [1, 1, 1, 1, 2])
    weights, metadata = load_sampling_weights(path, rows)
    assert weights.dtype == torch.float64
    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0)
    assert weights.sum().item() == pytest.approx(1)
    assert weights[:3].sum().item() == pytest.approx(0.5)
    assert weights[3:].sum().item() == pytest.approx(0.5)
    # Two copies together receive the same mass as one equally difficult scene.
    assert (weights[0] + weights[1]).item() == pytest.approx(weights[2].item())
    assert weights[4].item() == pytest.approx(2 * weights[3].item())
    assert metadata["sample_weights_source"] == str(path.resolve())
    assert metadata["sample_weights_source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    policy = metadata["sampler_policy"]
    assert policy["replacement"] is True
    assert policy["num_samples"] == len(rows)
    assert policy["duplicate_groups_discounted"] == 1
    assert policy["normalized_class_mass"] == pytest.approx({"cat": 0.5, "dog": 0.5})


def test_sampling_follows_manifest_order_independent_of_csv_order(rows, tmp_path):
    first = write_weights(tmp_path / "ordered.csv", rows, [1, 2, 1.5, 1, 2])
    second = write_weights(tmp_path / "reversed.csv", rows[::-1], [2, 1, 1.5, 2, 1])
    original, _ = load_sampling_weights(first, rows)
    reordered, _ = load_sampling_weights(second, rows)
    assert torch.equal(original, reordered)


def test_seeded_replacement_sampler_is_deterministic_across_epochs(rows, tmp_path):
    weights, _ = load_sampling_weights(write_weights(tmp_path / "weights.csv", rows), rows)
    dataset = TensorDataset(torch.arange(len(rows)))

    def make_loader():
        generator = torch.Generator().manual_seed(123)
        sampler = WeightedRandomSampler(weights, num_samples=len(rows), replacement=True,
                                        generator=generator)
        return DataLoader(dataset, batch_size=2, sampler=sampler, generator=generator, num_workers=0)

    first, second = make_loader(), make_loader()
    for _ in range(3):
        sampled_first = torch.cat([batch[0] for batch in first])
        sampled_second = torch.cat([batch[0] for batch in second])
        assert len(sampled_first) == len(rows)
        assert torch.equal(sampled_first, sampled_second)
        assert sampled_first.min() >= 0 and sampled_first.max() < len(rows)


@pytest.mark.parametrize("path", ["data/cat.val.jpg", "data/dog.test.jpg", "data/unknown.jpg"])
def test_csv_cannot_include_validation_test_or_unknown_paths(rows, tmp_path, path):
    outside = {"relative_path": path}
    csv_path = write_weights(tmp_path / "weights.csv", rows + [outside])
    with pytest.raises(ValueError, match="outside the current train split"):
        load_sampling_weights(csv_path, rows)


def test_csv_must_cover_all_train_paths(rows, tmp_path):
    path = write_weights(tmp_path / "weights.csv", rows[:-1])
    with pytest.raises(ValueError, match="cover every training image"):
        load_sampling_weights(path, rows)


def test_csv_rejects_repeated_training_path(rows, tmp_path):
    path = write_weights(tmp_path / "weights.csv", rows + [rows[0]])
    with pytest.raises(ValueError, match="repeats training path"):
        load_sampling_weights(path, rows)


@pytest.mark.parametrize("bad_weight", ["nan", "inf", "-inf", -1, 0, 2.001, "not-a-number", None])
def test_csv_rejects_invalid_or_out_of_range_weights(rows, tmp_path, bad_weight):
    path = write_weights(tmp_path / "weights.csv", rows, [bad_weight, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="weight"):
        load_sampling_weights(path, rows)


@pytest.mark.parametrize("split", ["val", "test"])
def test_helper_rejects_non_training_manifest_rows_even_if_csv_matches(rows, tmp_path, split):
    altered = [dict(row) for row in rows]
    altered[0]["split"] = split
    path = write_weights(tmp_path / "weights.csv", altered)
    with pytest.raises(ValueError, match="exclusively to train"):
        load_sampling_weights(path, altered)


def test_sampling_rejects_conflicting_class_labels_in_duplicate_group(rows, tmp_path):
    altered = [dict(row) for row in rows]
    altered[1]["label"] = 1
    path = write_weights(tmp_path / "weights.csv", altered)
    with pytest.raises(ValueError, match="conflicting class labels"):
        load_sampling_weights(path, altered)


def test_extreme_tiny_priorities_rejected_if_group_discount_underflows(rows, tmp_path):
    path = write_weights(tmp_path / "weights.csv", rows, [5e-324, 5e-324, 5e-324, 1, 1])
    with pytest.raises(ValueError, match="too extreme"):
        load_sampling_weights(path, rows)
