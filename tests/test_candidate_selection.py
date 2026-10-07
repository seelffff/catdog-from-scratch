import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/select_candidate.py"
spec = importlib.util.spec_from_file_location("select_candidate", SCRIPT)
selection_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection_script)


def candidate(root, name, accuracy, nll, ensemble=False, reports_directory=None):
    checkpoint = root / "artifacts/candidates" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"synthetic weights for {name}".encode())
    report = {
        "model": {"architecture": {"name": "logit_ensemble" if ensemble else "tiny_resnet18"},
                  "pretrained": False, "image_size": 224},
        "dataset": {"manifest_sha256": "a" * 64},
        "selection": {"source_checkpoint": checkpoint.relative_to(root).as_posix(), "source_epoch": 1,
                      "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      "selection_split": "val"},
        "validation": {"balanced_accuracy": accuracy, "negative_log_likelihood": nll},
    }
    path = (reports_directory or root / "reports") / "validation" / name / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return checkpoint, path


def test_ranking_uses_balanced_accuracy_then_nll_and_records_hashes(tmp_path):
    candidate(tmp_path, "lower", 0.94, 0.1)
    candidate(tmp_path, "tied_worse_nll", 0.95, 0.2)
    chosen, report_path = candidate(tmp_path, "tied_better_nll", 0.95, 0.15)
    result = selection_script.select_candidate(tmp_path)
    assert result["chosen_checkpoint"] == chosen.relative_to(tmp_path).as_posix()
    assert [item["rank"] for item in result["ranking"]] == [1, 2, 3]
    assert [item["balanced_accuracy"] for item in result["ranking"]] == [0.95, 0.95, 0.94]
    assert result["ranking"][0]["validation_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert result["chosen_checkpoint_sha256"] == hashlib.sha256(chosen.read_bytes()).hexdigest()
    assert result["test_accessed"] is False
    assert json.loads((tmp_path / "reports/model_selection.json").read_text())["chosen_kind"] == "single"


@pytest.mark.parametrize("ensemble_accuracy,accepted", [(0.9529, False), (0.953, True), (0.957, True), (0.949, False)])
def test_ensemble_requires_the_predeclared_minimum_gain(tmp_path, ensemble_accuracy, accepted):
    single, _ = candidate(tmp_path, "single", 0.95, 0.18)
    ensemble, _ = candidate(tmp_path, "ensemble", ensemble_accuracy, 0.16, ensemble=True)
    result = selection_script.select_candidate(tmp_path)
    assert result["ensemble_accepted"] is accepted
    assert result["chosen_checkpoint"] == (ensemble if accepted else single).relative_to(tmp_path).as_posix()
    assert result["rules"]["ensemble_min_gain"] == 0.003


def test_source_hash_mismatch_stops_selection_before_creating_result(tmp_path):
    checkpoint, _ = candidate(tmp_path, "single", 0.95, 0.2)
    checkpoint.write_bytes(b"updated weights after validation")
    with pytest.raises(ValueError, match="SHA-256"):
        selection_script.select_candidate(tmp_path)
    assert not (tmp_path / "reports/model_selection.json").exists()


def test_existing_final_test_report_prevents_reselection_without_reading_it(tmp_path, monkeypatch):
    candidate(tmp_path, "single", 0.95, 0.2)
    evaluation = tmp_path / "reports/evaluation.json"
    evaluation.write_bytes(b"must not be opened or parsed")
    monkeypatch.setattr(selection_script, "collect_candidates", lambda root: pytest.fail("Reports read after final test existed"))
    with pytest.raises(FileExistsError, match="test"):
        selection_script.select_candidate(tmp_path)
    assert evaluation.read_bytes() == b"must not be opened or parsed"
    assert not (tmp_path / "reports/model_selection.json").exists()


def test_legacy_report_without_source_hash_is_explicitly_excluded(tmp_path):
    candidate(tmp_path, "current", 0.95, 0.2)
    _, legacy = candidate(tmp_path, "legacy", 0.99, 0.05)
    report = json.loads(legacy.read_text())
    del report["selection"]["source_checkpoint_sha256"]
    legacy.write_text(json.dumps(report))
    result = selection_script.select_candidate(tmp_path)
    assert len(result["ranking"]) == 1
    assert len(result["excluded_reports"]) == 1
    assert "source_checkpoint_sha256" in result["excluded_reports"][0]["reason"]


def test_ensemble_only_reports_cannot_bypass_single_comparison(tmp_path):
    candidate(tmp_path, "ensemble", 0.95, 0.2, ensemble=True)
    with pytest.raises(ValueError, match="single"):
        selection_script.select_candidate(tmp_path)


def test_different_manifest_hashes_are_not_comparable(tmp_path):
    candidate(tmp_path, "first", 0.95, 0.2)
    _, other = candidate(tmp_path, "second", 0.96, 0.18)
    report = json.loads(other.read_text())
    report["dataset"]["manifest_sha256"] = "b" * 64
    other.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="разные разделения"):
        selection_script.select_candidate(tmp_path)


def test_v2_selection_preserves_closed_v1_reports(tmp_path):
    v1_reports = tmp_path / "reports"
    v1_reports.mkdir()
    evaluation = v1_reports / "evaluation.json"
    evaluation.write_bytes(b"V1 test is already closed")
    v1_selection = v1_reports / "model_selection.json"
    v1_selection.write_bytes(b"V1 choice remains frozen")
    v2_reports = v1_reports / "v2"
    chosen, _ = candidate(tmp_path, "v2_model", 0.97, 0.1, reports_directory=v2_reports)
    result = selection_script.select_candidate(tmp_path, "reports/v2")
    assert result["chosen_checkpoint"] == chosen.relative_to(tmp_path).as_posix()
    assert result["reports_directory"] == "reports/v2"
    assert (v2_reports / "model_selection.json").is_file()
    assert evaluation.read_bytes() == b"V1 test is already closed"
    assert v1_selection.read_bytes() == b"V1 choice remains frozen"


def test_v2_final_test_report_blocks_v2_reselection(tmp_path, monkeypatch):
    v2_reports = tmp_path / "reports/v2"
    candidate(tmp_path, "v2_model", 0.97, 0.1, reports_directory=v2_reports)
    (v2_reports / "evaluation.json").write_bytes(b"Do not read the closed V2 test")
    monkeypatch.setattr(selection_script, "collect_candidates", lambda *args: pytest.fail("Validation read after V2 test was closed"))
    with pytest.raises(FileExistsError, match="test"):
        selection_script.select_candidate(tmp_path, v2_reports)
    assert not (v2_reports / "model_selection.json").exists()


@pytest.mark.parametrize("use_symlink", [False, True])
def test_reports_directory_outside_project_is_rejected(tmp_path, use_symlink):
    root, outside = tmp_path / "project", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    if use_symlink:
        reports_directory = root / "reports-link"
        reports_directory.symlink_to(outside, target_is_directory=True)
    else:
        reports_directory = outside
    with pytest.raises(ValueError, match="Каталог отчётов"):
        selection_script.select_candidate(root, reports_directory)
    assert not (outside / "model_selection.json").exists()
