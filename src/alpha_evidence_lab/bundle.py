"""Import validated research bundles without depending on the other project's code."""

import hashlib
import json
from pathlib import Path

from .data import PricePath, load_csv


def load_bundle(directory: Path) -> tuple[PricePath, ...]:
    root = directory.resolve()
    raw = (root / "bundle.json").read_bytes()
    metadata = json.loads(raw)
    if metadata.get("schema") != "research_bundle_v1":
        raise ValueError("Unsupported research bundle version")
    if metadata.get("visibility") != "close_assumed":
        raise ValueError("Alpha requires close_assumed; delayed data are not silently backdated")
    source = str(metadata.get("source", "")).strip()
    if not source or not metadata.get("alpha_files"):
        raise ValueError("Bundle requires source and nonempty file inventory")
    source += "; bundle_sha256=" + hashlib.sha256(raw).hexdigest()
    paths = []
    seen: set[str] = set()
    for item in metadata["alpha_files"]:
        path = (root / item["file"]).resolve()
        if path.parent != root / "alpha" or path.suffix != ".csv" or path.stem in seen:
            raise ValueError("Unsafe or duplicate bundle path")
        seen.add(path.stem)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Bundle CSV hash mismatch")
        paths.append(load_csv(path, path.stem, source))
    return tuple(paths)
