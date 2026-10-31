"""Content fingerprint of a dataset directory. Standard library only.

    python3 - <root> < lab/core/dataset_manifest.py

The manifest is the sorted list of [relpath, size, sha256] for every regular
file under the root; the fingerprint is the sha256 of its canonical JSON. A
symlink anywhere in the tree is an error: rsync would copy it as a link, and a
link can point outside the root.

The GPU host computes its side by running this file from stdin over SSH, so
it must stay importable with no repo on sys.path and must not depend on the
host's checkout. The run prints one JSON object: root, files, fingerprint,
total_bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

CHUNK = 1 << 20


class ManifestError(RuntimeError):
    pass


def _files(root):
    """(relpath, abspath) for every regular file, sorted; refuses symlinks and specials."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                raise ManifestError(f"symlink in dataset: {os.path.relpath(path, root)}")
        dirnames.sort()
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):
                raise ManifestError(f"not a regular file: {os.path.relpath(path, root)}")
            out.append((os.path.relpath(path, root).replace(os.sep, "/"), path))
    return sorted(out)


def listing(root):
    """[relpath, size, mtime_ns] per file: cheap, used to decide whether a cached manifest is stale."""
    if not os.path.isdir(root):
        raise ManifestError(f"not a directory: {root}")
    rows = []
    for rel, path in _files(root):
        st = os.stat(path)
        rows.append([rel, st.st_size, st.st_mtime_ns])
    return rows


def manifest(root, progress=None):
    """[[relpath, size, sha256], ...] sorted by relpath. progress(done_bytes, total_bytes) is optional."""
    if not os.path.isdir(root):
        raise ManifestError(f"not a directory: {root}")
    files = _files(root)
    total = sum(os.path.getsize(path) for _, path in files)
    done = 0
    entries = []
    for rel, path in files:
        digest = hashlib.sha256()
        size = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
        entries.append([rel, size, digest.hexdigest()])
    if progress is not None:
        progress(total, total)
    return entries


def fingerprint(entries):
    canonical = json.dumps(entries, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def describe(root, entries):
    return {"root": str(root), "files": entries, "fingerprint": fingerprint(entries),
            "total_bytes": sum(e[1] for e in entries)}


def cached_manifest(root, cache_file, progress=None):
    """describe(root, manifest(root)), reusing cache_file while the (relpath, size, mtime) listing is unchanged."""
    rows = listing(root)
    key = hashlib.sha256(json.dumps([str(root), rows]).encode("utf-8")).hexdigest()
    cached = read_cache(cache_file)
    if cached is not None and cached.get("listing_key") == key:
        return {k: cached[k] for k in ("root", "files", "fingerprint", "total_bytes")}
    result = describe(root, manifest(root, progress))
    if listing(root) != rows:  # changed while hashing: the result is not for `key`
        return result
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    tmp = f"{cache_file}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({**result, "listing_key": key}, f)
    os.replace(tmp, cache_file)
    return result


def valid_cached_manifest(root, cache_file):
    """The cached result if it is still valid, else None. Never hashes."""
    try:
        rows = listing(root)
    except (OSError, ManifestError):
        return None
    key = hashlib.sha256(json.dumps([str(root), rows]).encode("utf-8")).hexdigest()
    cached = read_cache(cache_file)
    if cached is None or cached.get("listing_key") != key:
        return None
    return {k: cached[k] for k in ("root", "files", "fingerprint", "total_bytes")}


def read_cache(cache_file):
    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def main(argv):
    if len(argv) != 1:
        print("usage: python3 - <root> < dataset_manifest.py", file=sys.stderr)
        return 2
    try:
        print(json.dumps(describe(argv[0], manifest(argv[0]))))
    except (OSError, ManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
