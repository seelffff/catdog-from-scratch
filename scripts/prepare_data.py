"""Download the public dataset, safely extract it and create audited splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_LINK = "https://disk.yandex.ru/d/VzV5KXL8Y7Vvgg"
PUBLIC_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
MAX_ARCHIVE_BYTES = 5 * 1024**3
MAX_EXTRACTED_BYTES = 4 * 1024**3
MAX_MEMBER_BYTES = 30 * 1024**2
MAX_FILES = 50_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validated_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_FILES:
        raise ValueError("В архиве слишком много файлов")
    seen = set()
    total = 0
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if (not name or "\\" in name or "\x00" in name or path.is_absolute()
                or any(part in {"..", "."} for part in name.rstrip("/").split("/"))
                or ":" in name or path.parts[0] != "train"):
            raise ValueError(f"Небезопасный или неожиданный путь в архиве: {name!r}")
        normalized = path.as_posix().casefold()
        if normalized in seen:
            raise ValueError(f"Повторяющийся путь в архиве: {name!r}")
        seen.add(normalized)
        file_type = stat.S_IFMT(member.external_attr >> 16)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError(f"Ссылки и специальные файлы в архиве запрещены: {name!r}")
        if member.flag_bits & 1:
            raise ValueError("Зашифрованные файлы не поддерживаются")
        if not member.is_dir():
            if len(path.parts) != 2:
                raise ValueError("Фотографии должны лежать непосредственно в train/")
            if member.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"Слишком большой файл в архиве: {name!r}")
            if member.file_size > 1024 * 1024 and member.file_size > max(member.compress_size, 1) * 1000:
                raise ValueError("Подозрительный коэффициент сжатия ZIP")
            total += member.file_size
    if total > MAX_EXTRACTED_BYTES:
        raise ValueError("Распакованный архив превышает лимит 4 ГБ")
    if not any(not member.is_dir() for member in members):
        raise ValueError("В архиве нет фотографий")
    return members


def download_archive(public_link: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"public_key": public_link})
    request = urllib.request.Request(PUBLIC_API + "?" + query, headers={"User-Agent": "catsdogs-scratch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.loads(response.read(1024 * 1024))
    href = metadata.get("href", "")
    parsed = urllib.parse.urlparse(href)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Yandex API не вернул корректную HTTPS-ссылку")
    request = urllib.request.Request(href, headers={"User-Agent": "catsdogs-scratch/1.0"})
    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="train-", suffix=".zip.part", delete=False) as output:
            partial = Path(output.name)
            with urllib.request.urlopen(request, timeout=120) as response:
                expected_size = response.headers.get("Content-Length")
                if expected_size is not None and int(expected_size) > MAX_ARCHIVE_BYTES:
                    raise ValueError("Архив превышает лимит 5 ГБ")
                downloaded, started, last_update = 0, time.monotonic(), time.monotonic()
                while block := response.read(1024 * 1024):
                    downloaded += len(block)
                    if downloaded > MAX_ARCHIVE_BYTES or time.monotonic() - started > 1800:
                        raise ValueError("Превышен размер архива или 30 минут скачивания")
                    output.write(block)
                    if time.monotonic() - last_update >= 10:
                        print(f"Скачано {downloaded / 1024**2:.1f} МБ", flush=True)
                        last_update = time.monotonic()
                if expected_size is not None and downloaded != int(expected_size):
                    raise ValueError("Архив скачан не полностью")
        with zipfile.ZipFile(partial) as archive:
            validated_members(archive)
        os.replace(partial, destination)
        partial = None
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def extract_archive(archive_path: Path, raw_dir: Path) -> dict:
    """Stage extraction first; never overwrite an existing train directory."""
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Архив превышает лимит 5 ГБ")
    with zipfile.ZipFile(archive_path) as archive:
        members = validated_members(archive)
        files = [member for member in members if not member.is_dir()]
        if raw_dir.is_symlink() or raw_dir.parent.is_symlink():
            raise ValueError("Папки для распаковки не должны быть символическими ссылками")
        target = raw_dir / "train"
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise ValueError("data/raw/train должна быть обычной папкой")
            expected = {PurePosixPath(member.filename).name: member.file_size for member in files}
            actual = {path.name: path.stat().st_size for path in target.iterdir() if path.is_file() and not path.is_symlink()}
            if actual != expected or any(path.is_symlink() for path in target.iterdir()):
                raise FileExistsError("Папка train уже существует и отличается от архива. Используйте новую папку проекта; фотографии не перезаписаны.")
            for member in files:
                checksum = 0
                with (target / PurePosixPath(member.filename).name).open("rb") as image_file:
                    while block := image_file.read(1024 * 1024):
                        checksum = zlib.crc32(block, checksum)
                if checksum != member.CRC:
                    raise FileExistsError("Содержимое существующей фотографии отличается от архива; фотографии не перезаписаны.")
            status = "existing_images_preserved"
        else:
            raw_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=raw_dir.parent, prefix="dataset-extract-") as directory:
                staging = Path(directory)
                for member in members:
                    destination = staging.joinpath(*PurePosixPath(member.filename).parts)
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                raw_dir.mkdir(parents=True, exist_ok=True)
                (staging / "train").rename(target)
            status = "extracted"
    return {"status": status, "archive_files": len(files), "uncompressed_bytes": sum(member.file_size for member in files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--archive", type=Path, help="Существующий ZIP для подготовки без скачивания")
    parser.add_argument("--public-link", default=PUBLIC_LINK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-audit", action="store_true", help="Только скачать и распаковать")
    args = parser.parse_args()
    root = args.project_root.resolve()
    archive_path = args.archive.resolve() if args.archive else root / "data/source/train.zip"
    try:
        if not archive_path.exists():
            if args.archive:
                raise FileNotFoundError(archive_path)
            print("Скачиваем публичный датасет Yandex Disk…", flush=True)
            download_archive(args.public_link, archive_path)
        result = extract_archive(archive_path, root / "data/raw")
        source_record = {
            "public_link": args.public_link,
            "archive_sha256": sha256_file(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        report_path = root / "reports/dataset_source.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(source_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.skip_audit:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from catsdogs.data import audit_dataset

            audit = audit_dataset(root / "data/raw/train", root / "data/splits.csv", root / "artifacts/data_audit.json",
                                  project_root=root, seed=args.seed, workers=args.workers)
            source_record["split_counts"] = audit["split_counts"]
            source_record["invalid_images"] = len(audit["invalid_images"])
        print(json.dumps(source_record, ensure_ascii=False, indent=2), flush=True)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Ошибка подготовки данных: {exc}\n")


if __name__ == "__main__":
    main()
