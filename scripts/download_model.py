"""Download the portable published checkpoint and verify its exact SHA-256."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "models/model_manifest.json").read_text())
    target = ROOT / "models" / manifest["filename"]
    expected = manifest["sha256"]
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("Existing model has a different checksum; preserve it before downloading another version.")
        print("Model already exists and its checksum matches.")
        return
    request = urllib.request.Request(manifest["download_url"], headers={"User-Agent": "catdog-from-scratch/1.0"})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > manifest["bytes"]:
                        raise ValueError("Downloaded model is larger than the published checkpoint.")
                    digest.update(block)
                    handle.write(block)
            if total != manifest["bytes"] or digest.hexdigest() != expected:
                raise ValueError("Downloaded model checksum or size differs from the manifest.")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved verified model: {target.name}")


if __name__ == "__main__":
    main()
