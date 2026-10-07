"""Freeze a validation-only choice before a final test report exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_MIN_GAIN = Decimal("0.003")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_reports_directory(root: Path, reports_directory: Path | str | None = None) -> Path:
    root = root.resolve()
    directory = Path(reports_directory) if reports_directory is not None else root / "reports"
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Каталог отчётов должен находиться внутри проекта")
    return directory


def assert_before_test(root: Path, reports_directory: Path | str | None = None) -> None:
    evaluation = resolve_reports_directory(root, reports_directory) / "evaluation.json"
    if evaluation.exists() or evaluation.is_symlink():
        raise FileExistsError("Финальная оценка test уже существует: переизбирать модель запрещено")


def collect_candidates(root: Path, reports_directory: Path | str | None = None) -> tuple[list[dict], list[dict]]:
    """Read report metadata and checkpoint bytes, never images or test metrics."""
    root = root.resolve()
    reports_directory = resolve_reports_directory(root, reports_directory)
    ranked, excluded = [], []
    paths = sorted((reports_directory / "validation").glob("*/validation.json"))
    if not paths:
        raise ValueError(f"Не найдены {reports_directory.relative_to(root)}/validation/*/validation.json")
    manifest_hash = None
    for report_path in paths:
        payload = report_path.read_bytes()
        report = json.loads(payload)
        selection = report["selection"]
        expected_hash = selection.get("source_checkpoint_sha256")
        if expected_hash is None:
            excluded.append({"report": report_path.relative_to(root).as_posix(),
                             "reason": "Legacy report has no source_checkpoint_sha256; regenerate validation to make it eligible"})
            continue
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"Некорректный SHA-256 в {report_path}")
        checkpoint = Path(selection["source_checkpoint"])
        checkpoint = checkpoint.resolve() if checkpoint.is_absolute() else (root / checkpoint).resolve()
        if not checkpoint.is_relative_to(root):
            raise ValueError("Источник checkpoint должен находиться внутри проекта")
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 источника не совпадает с validation: {checkpoint}")
        if selection.get("selection_split") != "val" or "test" in report:
            raise ValueError("Для выбора допускаются только validation-only отчёты")
        if report["model"].get("pretrained") is not False:
            raise ValueError("Кандидат должен быть собственной scratch-моделью")
        current_manifest_hash = report["dataset"]["manifest_sha256"]
        if not isinstance(current_manifest_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", current_manifest_hash):
            raise ValueError("Отчёт должен содержать SHA-256 фиксированного разделения")
        if manifest_hash is not None and current_manifest_hash != manifest_hash:
            raise ValueError("Validation-кандидаты используют разные разделения датасета")
        manifest_hash = current_manifest_hash
        validation = report["validation"]
        accuracy = float(validation["balanced_accuracy"])
        nll = float(validation["negative_log_likelihood"])
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or not math.isfinite(nll) or nll < 0:
            raise ValueError("Некорректные validation-метрики")
        architecture = report["model"]["architecture"]
        ranked.append({
            "checkpoint": checkpoint.relative_to(root).as_posix(),
            "source_checkpoint_sha256": actual_hash,
            "validation_report": report_path.relative_to(root).as_posix(),
            "validation_report_sha256": hashlib.sha256(payload).hexdigest(),
            "kind": "ensemble" if architecture["name"] == "logit_ensemble" else "single",
            "architecture": architecture,
            "image_size": report["model"]["image_size"],
            "source_epoch": selection.get("source_epoch"),
            "balanced_accuracy": accuracy,
            "negative_log_likelihood": nll,
            "manifest_sha256": manifest_hash,
            "tta_horizontal_flip": selection.get("tta_horizontal_flip", False),
            "own_training_provenance": selection.get("own_training_provenance"),
        })
    ranked.sort(key=lambda item: (-item["balanced_accuracy"], item["negative_log_likelihood"], item["validation_report"]))
    for index, item in enumerate(ranked, 1):
        item["rank"] = index
    return ranked, excluded


def select_candidate(root: Path, reports_directory: Path | str | None = None) -> dict:
    root = root.resolve()
    reports_directory = resolve_reports_directory(root, reports_directory)
    assert_before_test(root, reports_directory)
    ranking, excluded = collect_candidates(root, reports_directory)
    singles = [item for item in ranking if item["kind"] == "single"]
    ensembles = [item for item in ranking if item["kind"] == "ensemble"]
    if not singles:
        raise ValueError("Нужен хотя бы один проверенный single-кандидат для сравнения")
    best_single = singles[0]
    best_ensemble = ensembles[0] if ensembles else None
    gain = (Decimal(str(best_ensemble["balanced_accuracy"])) - Decimal(str(best_single["balanced_accuracy"]))) if best_ensemble else None
    ensemble_accepted = gain is not None and gain >= ENSEMBLE_MIN_GAIN
    chosen = best_ensemble if ensemble_accepted else best_single
    result = {
        "selection_split": "val",
        "test_accessed": False,
        "reports_directory": reports_directory.relative_to(root).as_posix(),
        "rules": {"ranking_primary": "validation.balanced_accuracy descending",
                  "ranking_tie_break": "validation.negative_log_likelihood ascending",
                  "ensemble_min_gain": float(ENSEMBLE_MIN_GAIN),
                  "ensemble_min_gain_percentage_points": float(ENSEMBLE_MIN_GAIN * 100),
                  "ensemble_acceptance": "Best ensemble is selected only if balanced-accuracy gain over best single is >=0.003",
                  "final_test_guard": f"No reselection after {reports_directory.relative_to(root).as_posix()}/evaluation.json exists"},
        "chosen_checkpoint": chosen["checkpoint"],
        "chosen_checkpoint_sha256": chosen["source_checkpoint_sha256"],
        "chosen_kind": chosen["kind"],
        "manifest_sha256": chosen["manifest_sha256"],
        "best_single": best_single,
        "best_ensemble": best_ensemble,
        "ensemble_gain": float(gain) if gain is not None else None,
        "ensemble_accepted": ensemble_accepted,
        "ranking": ranking,
        "excluded_reports": excluded,
    }
    assert_before_test(root, reports_directory)
    output = reports_directory / "model_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix="model-selection-", suffix=".json.part", delete=False) as file:
            temporary = Path(file.name)
            file.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        assert_before_test(root, reports_directory)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--reports-directory", type=Path,
                        help="Каталог отчётов внутри проекта; по умолчанию reports/ (для V2: reports/v2)")
    args = parser.parse_args()
    try:
        print(json.dumps(select_candidate(args.project_root, args.reports_directory), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Ошибка выбора модели: {exc}\n")


if __name__ == "__main__":
    main()
