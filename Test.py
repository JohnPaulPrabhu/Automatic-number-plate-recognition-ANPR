# cache_profiler.py
from __future__ import annotations
import hashlib, json, gzip, os, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Iterable, Literal

HashMode = Literal["mtime+size", "sha256"]

@dataclass
class FileEntry:
    rel_path: str
    size: int
    mtime_ns: int
    sha256: Optional[str] = None

@dataclass
class Manifest:
    dataset_root: str
    hash_mode: HashMode
    files: List[FileEntry]

@dataclass
class CacheBlob:
    version: int
    created_at: float
    dataset_manifest_hash: str
    params_hash: str
    profiler_result: Dict

CACHE_VERSION = 1

def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _normalize_root(root: Path) -> Path:
    return root.expanduser().resolve()

def _iter_files(
    root: Path,
    include_exts: Optional[Iterable[str]] = None,
    ignore_hidden: bool = True,
) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if ignore_hidden and (name.startswith(".") or any(part.startswith(".") for part in p.relative_to(root).parts)):
            continue
        if include_exts:
            if p.suffix.lower() not in {e.lower() for e in include_exts}:
                continue
        yield p

def build_manifest(
    dataset_root: str | Path,
    hash_mode: HashMode = "mtime+size",
    include_exts: Optional[Iterable[str]] = None,
    ignore_hidden: bool = True,
) -> Manifest:
    root = _normalize_root(Path(dataset_root))
    files: List[FileEntry] = []
    for p in _iter_files(root, include_exts=include_exts, ignore_hidden=ignore_hidden):
        stat = p.stat()
        entry = FileEntry(
            rel_path=str(p.relative_to(root)),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=_sha256_file(p) if hash_mode == "sha256" else None,
        )
        files.append(entry)
    # Sort for deterministic hashing
    files.sort(key=lambda e: e.rel_path)
    return Manifest(dataset_root=str(root), hash_mode=hash_mode, files=files)

def hash_manifest(m: Manifest) -> str:
    # Only include stable fields for the hash
    pieces = [m.dataset_root, m.hash_mode]
    for f in m.files:
        pieces.append(f.rel_path)
        pieces.append(str(f.size))
        pieces.append(str(f.mtime_ns))
        if m.hash_mode == "sha256":
            pieces.append(f.sha256 or "")
    return _hash_str("|".join(pieces))

def diff_manifests(old: Manifest, new: Manifest) -> Dict[str, List[str]]:
    old_map = {f.rel_path: f for f in old.files}
    new_map = {f.rel_path: f for f in new.files}
    old_set, new_set = set(old_map), set(new_map)
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    changed = []
    intersect = old_set & new_set
    for rel in intersect:
        a, b = old_map[rel], new_map[rel]
        # consider modified if size/mtime differ (and sha256 if present)
        if (a.size != b.size) or (a.mtime_ns != b.mtime_ns) or ((a.sha256 or "") != (b.sha256 or "")):
            changed.append(rel)
    return {"added": added, "removed": removed, "modified": sorted(changed)}

def _read_cache(cache_path: str | Path) -> Optional[Tuple[CacheBlob, Manifest]]:
    p = Path(cache_path)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        blob = CacheBlob(
            version=data["version"],
            created_at=data["created_at"],
            dataset_manifest_hash=data["dataset_manifest_hash"],
            params_hash=data["params_hash"],
            profiler_result=data["profiler_result"],
        )
        m = data["manifest"]
        manifest = Manifest(
            dataset_root=m["dataset_root"],
            hash_mode=m["hash_mode"],
            files=[FileEntry(**fe) for fe in m["files"]],
        )
        return blob, manifest
    except Exception:
        return None  # treat as no cache if corrupted

def _write_cache(cache_path: str | Path, blob: CacheBlob, manifest: Manifest) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": blob.version,
        "created_at": blob.created_at,
        "dataset_manifest_hash": blob.dataset_manifest_hash,
        "params_hash": blob.params_hash,
        "profiler_result": blob.profiler_result,
        "manifest": {
            "dataset_root": manifest.dataset_root,
            "hash_mode": manifest.hash_mode,
            "files": [asdict(f) for f in manifest.files],
        },
    }
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _hash_params(params: Optional[Dict]) -> str:
    if not params:
        return _hash_str("none")
    # stable JSON
    return _hash_str(json.dumps(params, sort_keys=True, separators=(",", ":")))

def load_or_profile(
    dataset_root: str | Path,
    cache_path: str | Path,
    run_profiler_fn: Callable[[str], Dict],
    *,
    hash_mode: HashMode = "mtime+size",
    include_exts: Optional[Iterable[str]] = None,
    ignore_hidden: bool = True,
    params: Optional[Dict] = None,
    force_reprofile: bool = False,
    delete_when_changed: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Main entry point:
      - Checks cache; if valid, returns cached profiler_result
      - If dataset changed or params changed (or forced), re-runs profiler, writes cache, returns result

    Args:
        run_profiler_fn: function that accepts dataset_root (str) and returns a Python dict (your profiler result)
        params: optional dict of settings that also invalidate cache when changed (e.g., quant config)
        hash_mode: "mtime+size" (fast) or "sha256" (strict)
    """
    dataset_root = str(_normalize_root(Path(dataset_root)))
    params_hash = _hash_params(params)
    existing = _read_cache(cache_path)

    # Build *current* manifest
    current_manifest = build_manifest(
        dataset_root, hash_mode=hash_mode, include_exts=include_exts, ignore_hidden=ignore_hidden
    )
    current_manifest_hash = hash_manifest(current_manifest)

    # Use cache if exists and valid
    if not force_reprofile and existing:
        blob, old_manifest = existing

        # Safety on version mismatch: reprofile
        if blob.version == CACHE_VERSION and old_manifest.dataset_root == dataset_root:
            changed = diff_manifests(old_manifest, current_manifest)
            changed_any = any(changed.values())

            if verbose:
                print(f"[cache] cache exists: {cache_path}")
                if changed_any:
                    print(f"[cache] dataset changed → {changed}")
                if blob.params_hash != params_hash:
                    print("[cache] params changed → invalidating cache")

            if (not changed_any) and (blob.params_hash == params_hash) and (blob.dataset_manifest_hash == current_manifest_hash):
                if verbose:
                    print("[cache] using cached profiler results")
                return blob.profiler_result
            else:
                # invalidate cache on change
                if delete_when_changed and Path(cache_path).exists():
                    try:
                        os.remove(cache_path)
                        if verbose:
                            print("[cache] invalidated cache file removed")
                    except Exception as e:
                        if verbose:
                            print(f"[cache] warning: failed to remove cache: {e}")

    # Run profiler fresh
    if verbose:
        print("[cache] running profiler…")
    result = run_profiler_fn(dataset_root)  # <-- your heavy profiler

    # Save new cache
    blob = CacheBlob(
        version=CACHE_VERSION,
        created_at=time.time(),
        dataset_manifest_hash=current_manifest_hash,
        params_hash=params_hash,
        profiler_result=result,
    )
    _write_cache(cache_path, blob, current_manifest)
    if verbose:
        print(f"[cache] profiler results cached → {cache_path}")
    return result
