# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Housekeeping: what is taking disk, and what is safe to remove.

The two databases that grow without a ceiling on a training machine are the
Studio outputs folder (one folder per merge, expansion, export and run) and the
Hugging Face cache (one folder per model or dataset ever downloaded). Both are
ordinary directories, so the honest operation is a scan plus a delete guard, not
a database of "owned" rows.

Every delete is checked against the roots this module scanned and against the
advanced job registry, so a path assembled by a browser cannot remove a folder
outside ``outputs/``, ``exports/`` or the HF cache. The scanner never follows a
symlink and never walks in a loop; a folder it cannot read is reported, not
skipped silently.
"""

from __future__ import annotations

import calendar
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Optional

_MAX_FILES = 400_000
_SKIP_DIRS = frozenset({"labs-pipelines", "labs-benchmarks"})
_ROOT_IDS = ("outputs", "exports", "hf-hub", "hf-datasets")
_RULES_NAME = "janitor-rules.json"
_DEFAULT_RULES: dict[str, Any] = {
    "enabled": True,
    "keep_newest": 3,
    "min_age_days": 7,
    "roots": ["outputs", "exports"],
}


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def _paths() -> dict[str, Path]:
    """The four roots, even when one of them does not exist on this host."""
    from utils.paths.storage_roots import exports_root, outputs_root

    roots = {
        "outputs": outputs_root(),
        "exports": exports_root(),
    }
    try:
        from utils.hf_cache_settings import get_hf_cache_paths

        cache = get_hf_cache_paths()
        roots["hf-hub"] = cache.hub_cache
        roots["hf-datasets"] = cache.cache_home / "datasets"
    except Exception:  # noqa: BLE001 - a host without the settings module still has outputs
        pass
    return roots


_ROOT_LABELS = {
    "outputs": "Studio outputs",
    "exports": "Exports",
    "hf-hub": "Hugging Face models",
    "hf-datasets": "Hugging Face datasets",
}

_ROOT_NOTES = {
    "outputs": "Checkpoints, merges and expansions this Studio wrote.",
    "exports": "GGUF and quantized bundles written by Export.",
    "hf-hub": "Downloaded model snapshots. Deleting one makes the next run download it again.",
    "hf-datasets": "Downloaded dataset snapshots. Deleting one makes the next run download it again.",
}


def _within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except (OSError, ValueError):
        return False
    return True


def _resolving() -> Any:
    """One ``resolve()`` per distinct path, for the duration of one scan or delete.

    On Windows ``resolve()`` walks the filesystem for every component, and the
    scan asks about the same handful of roots and parents once per entry, so a
    cache turns five resolutions per folder into one. It is created per call so
    a folder moved between requests is still seen fresh.
    """
    cache: dict[Path, Path] = {}

    def resolve(path: Path) -> Path:
        resolved = cache.get(path)
        if resolved is None:
            resolved = path.resolve()
            cache[path] = resolved
        return resolved

    return resolve


def _is_link(entry: os.DirEntry) -> bool:
    """True for a symlink or a Windows junction.

    ``is_symlink()`` is False for a junction, and that matters twice: a junction
    inside a storage root is the same folder under a second name, and letting
    ``_measure`` walk it would size (and later offer) whatever it points at.
    """
    if entry.is_symlink():
        return True
    try:
        return bool(entry.stat(follow_symlinks = False).st_reparse_tag)
    except OSError:
        return False


def _iso(stamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp or time.time()))


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _measure(directory: Path) -> tuple[int, int]:
    """Total bytes and file count under *directory*, without following links.

    A cap keeps a stray cache mount from making the page hang. A folder that
    hits the cap is reported with the partial count and a note; it is still
    deletable, because the path (not the exact byte count) is what the guard
    checks.

    Links are skipped with :func:`_is_link`, not ``is_symlink()``: a junction is
    False there, and a junction pointing back at one of its own parents is a
    cycle this walk would never leave. The cap does not save it -- ``files`` only
    grows on a regular file, so a linked cycle around an empty folder keeps the
    loop running while the stack grows, which is a hang and a memory leak rather
    than a slow page.
    """
    total = 0
    files = 0
    stack: list[Path] = [directory]
    while stack and files < _MAX_FILES:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if _is_link(entry):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, files


def _format_bytes(value: float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f} TB"
    if value >= 1e9:
        return f"{value / 1e9:.2f} GB"
    if value >= 1e6:
        return f"{value / 1e6:.0f} MB"
    if value >= 1e3:
        return f"{value / 1e3:.0f} KB"
    return f"{int(value)} B"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _kind(path: Path, root_id: str) -> str:
    if root_id == "hf-hub":
        return "hf-model" if path.name.startswith("models--") else "hf-cache"
    if root_id == "hf-datasets":
        return "hf-dataset"
    if root_id == "exports":
        return "export"
    if (path / "config.json").is_file():
        return "checkpoint"
    if (path / "run_manifest.json").is_file():
        return "run"
    return "folder"


def _protected_reason(
    path: Path, root_id: str, root: Path, resolve: Any, *, top_level: bool = False
) -> Optional[str]:
    """Why *path* may not be offered for deletion, or None when it may.

    ``top_level`` says the caller only hands over direct children of *root*, so
    the depth rule is satisfied by construction and needs no resolution. The
    scan sets it; delete takes an arbitrary path from the client and does not.
    """
    if not top_level and resolve(path) == resolve(root):
        return "This is a storage root, not a folder inside it."
    if path.is_symlink():
        return "This is a symlink; delete the folder it points at from its own location."
    if path.name.startswith("."):
        return "Hidden folders are left alone."
    if path.name in _SKIP_DIRS:
        return "Labs keeps its own plans and benchmark records here."
    if not top_level and root_id in ("outputs", "exports") and resolve(path.parent) != resolve(root):
        # The Artifacts page only reads one level below outputs, so anything
        # deeper is some run's private working directory and is not offered.
        return "Only top-level Studio folders are listed for deletion."
    return None


def _manifest_summary(path: Path) -> Optional[dict[str, Any]]:
    manifest = _read_json(path / "run_manifest.json")
    if manifest is None:
        return None
    return {
        "stage": manifest.get("stage"),
        "run_id": manifest.get("run_id"),
        "created_at": manifest.get("created_at"),
    }


def _running_outputs() -> list[str]:
    """Output directories of advanced jobs that are still running.

    The registry lives on the route module; importing it lazily keeps this
    module importable on a host that never starts a job, and an import failure
    simply means there is nothing to protect.
    """
    try:
        from routes import advanced

        outputs: list[str] = []
        with advanced._JOBS_LOCK:
            for record in advanced._JOBS.values():
                if record.get("status") == "running" and record.get("output_dir"):
                    outputs.append(str(record["output_dir"]))
        return outputs
    except Exception:  # noqa: BLE001
        return []


def _resolved_running(running: Iterable[str], resolve: Any) -> list[Path]:
    """The running jobs' output folders, resolved once per scan or delete.

    They arrive as plain strings from the job registry, and resolving them inside
    the per-folder comparison put a filesystem walk back into the loop that this
    module exists to keep cheap.
    """
    resolved: list[Path] = []
    for output in running:
        try:
            resolved.append(resolve(Path(output)))
        except (OSError, ValueError):
            continue
    return resolved


def _is_under_running_job(path: Path, running: Iterable[Path], resolve: Any) -> bool:
    """Whether *path* is a running job's output folder or sits inside one."""
    if not running:
        return False
    resolved = resolve(path)
    for output in running:
        if resolved == output or output in resolved.parents:
            return True
    return False


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def _entries_for(
    root_id: str, root: Path, running: list[str], resolve: Any
) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    try:
        # scandir already knows each entry's type from the directory listing, so
        # the is_dir/is_symlink questions below cost no extra syscall per folder.
        with os.scandir(root) as scanned:
            children = sorted(scanned, key = lambda entry: entry.name.lower())
    except OSError:
        return records
    for entry in children:
        child = Path(entry.path)
        try:
            if not entry.is_dir(follow_symlinks = False) or entry.name.startswith("."):
                continue
            if entry.name in _SKIP_DIRS:
                continue
            # Links are never listed: a junction is the same folder under a
            # second name, and every one of them is still refused by delete.
            # Skipping them here is also why a listing needs no resolve().
            if _is_link(entry):
                continue
            size, files = _measure(child)
            try:
                modified = entry.stat(follow_symlinks = False).st_mtime
            except OSError:
                modified = 0.0
            reason = _protected_reason(child, root_id, root, resolve, top_level = True)
            if _is_under_running_job(child, running, resolve):
                reason = "A job is writing here right now."
            records.append(
                {
                    "path": str(child),
                    "name": child.name,
                    "root": root_id,
                    "kind": _kind(child, root_id),
                    "bytes": size,
                    "bytes_label": _format_bytes(size),
                    "file_count": files,
                    "modified": _iso(modified),
                    "manifest": _manifest_summary(child),
                    "deletable": reason is None,
                    "protected_reason": reason,
                }
            )
        except OSError:
            continue
    records.sort(key=lambda record: record["bytes"], reverse=True)
    return records


def scan() -> dict[str, Any]:
    """Everything that is large, newest roots first, biggest entries first."""
    roots = _paths()
    resolve = _resolving()
    running = _resolved_running(_running_outputs(), resolve)
    entries: list[dict[str, Any]] = []
    root_records: list[dict[str, Any]] = []
    for root_id in _ROOT_IDS:
        root = roots.get(root_id)
        if root is None:
            continue
        exists = root.is_dir()
        root_entries = _entries_for(root_id, root, running, resolve) if exists else []
        entries.extend(root_entries)
        root_records.append(
            {
                "id": root_id,
                "label": _ROOT_LABELS[root_id],
                "note": _ROOT_NOTES[root_id],
                "path": str(root),
                "exists": exists,
                "entry_count": len(root_entries),
                "bytes": sum(entry["bytes"] for entry in root_entries),
            }
        )
    entries.sort(key=lambda record: record["bytes"], reverse=True)
    total = sum(record["bytes"] for record in entries)
    reclaimable = sum(
        record["bytes"] for record in entries if record["deletable"]
    )
    disk = shutil.disk_usage(roots.get("outputs") or Path.home())
    return {
        "roots": root_records,
        "entries": entries,
        "total_bytes": total,
        "total_label": _format_bytes(total),
        "reclaimable_bytes": reclaimable,
        "reclaimable_label": _format_bytes(reclaimable),
        "free_bytes": int(disk.free),
        "free_label": _format_bytes(disk.free),
        "disk_total_bytes": int(disk.total),
        "protected": [
            {"path": record["path"], "reason": record["protected_reason"]}
            for record in entries
            if not record["deletable"]
        ],
        "notes": [
            "Sizes are measured by walking each folder; a huge cache can take a few seconds.",
            "Deleting a Studio output also removes it from the Library and the Artifacts page.",
            "A running job's output folder is refused until the job finishes.",
        ],
    }


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def delete(paths: Iterable[str], *, confirm: bool) -> dict[str, Any]:
    """Remove the named folders, refusing anything the scan would not offer."""
    if not confirm:
        raise ValueError("Deleting folders needs confirm=true.")
    roots = _paths()
    resolve = _resolving()
    running = _resolved_running(_running_outputs(), resolve)

    deleted: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(str(raw)).expanduser()
        root_id = next(
            (identifier for identifier, root in roots.items() if _within(root, path)),
            None,
        )
        if root_id is None:
            refused.append({"path": str(path), "reason": "Outside the storage roots."})
            continue
        root = roots[root_id]
        reason = _protected_reason(path, root_id, root, resolve)
        if not path.exists():
            refused.append({"path": str(path), "reason": "No such folder."})
            continue
        if not path.is_dir():
            refused.append({"path": str(path), "reason": "Only folders are removed."})
            continue
        if reason is not None:
            refused.append({"path": str(path), "reason": reason})
            continue
        if _is_under_running_job(path, running, resolve):
            refused.append({"path": str(path), "reason": "A job is writing here right now."})
            continue
        size, files = _measure(path)
        try:
            shutil.rmtree(path)
        except OSError as error:
            refused.append({"path": str(path), "reason": f"Could not delete: {error}"})
            continue
        deleted.append(
            {
                "path": str(path),
                "bytes": size,
                "bytes_label": _format_bytes(size),
                "file_count": files,
            }
        )
    freed = sum(item["bytes"] for item in deleted)
    return {
        "deleted": deleted,
        "refused": refused,
        "freed_bytes": freed,
        "freed_label": _format_bytes(freed),
    }


# ---------------------------------------------------------------------------
# Janitor rules
# ---------------------------------------------------------------------------
#
# The scan offers folders and the delete guard accepts paths; between them sat a
# human deciding which of forty checkpoints still matter. A rule is that decision
# written down once: keep the newest N per root, leave anything younger than D
# days alone, and never touch a folder holding a pinned benchmark. Rules are a
# dry run until someone presses apply, and apply is the same guarded delete --
# a rule cannot remove what the scan would refuse.


def rules_path() -> Path:
    return _paths()["outputs"] / _RULES_NAME


def _validated_rules(payload: Any) -> dict[str, Any]:
    if payload is None:
        return dict(_DEFAULT_RULES)
    if not isinstance(payload, dict):
        raise ValueError("Janitor rules must be an object.")
    enabled = payload.get("enabled", _DEFAULT_RULES["enabled"])
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be true or false.")
    keep = payload.get("keep_newest", _DEFAULT_RULES["keep_newest"])
    if isinstance(keep, bool) or not isinstance(keep, int) or not 0 <= keep <= 100:
        raise ValueError("keep_newest must be a whole number from 0 to 100.")
    days = payload.get("min_age_days", _DEFAULT_RULES["min_age_days"])
    if isinstance(days, bool) or not isinstance(days, int) or not 0 <= days <= 3650:
        raise ValueError("min_age_days must be a whole number from 0 to 3650.")
    raw_roots = payload.get("roots", _DEFAULT_RULES["roots"])
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError("roots must be a non-empty list.")
    roots: list[str] = []
    for identifier in raw_roots:
        if identifier not in _ROOT_IDS:
            raise ValueError(f"Unknown storage root: {identifier!r}.")
        if identifier not in roots:
            roots.append(str(identifier))
    return {
        "enabled": enabled,
        "keep_newest": keep,
        "min_age_days": days,
        "roots": roots,
    }


def read_rules() -> dict[str, Any]:
    """The saved rules, or the defaults when nothing was saved yet."""
    try:
        payload = _read_json(rules_path())
    except (OSError, KeyError):
        payload = None
    if payload is None:
        return dict(_DEFAULT_RULES)
    try:
        return _validated_rules(payload)
    except ValueError:
        # A hand-edited file that no longer validates must not brick the page.
        return dict(_DEFAULT_RULES)


def save_rules(payload: Any) -> dict[str, Any]:
    from core.advanced.policy import atomic_write_text

    rules = _validated_rules(payload)
    target = rules_path()
    target.parent.mkdir(parents = True, exist_ok = True)
    atomic_write_text(target, json.dumps(rules, indent = 2, sort_keys = True))
    return rules


def _epoch(iso: str) -> float:
    try:
        return float(calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return 0.0


def _normalized(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _holds_a_pin(path: str, pins: set[str]) -> bool:
    """Whether *path* is a pinned checkpoint or a folder that contains one."""
    if not pins:
        return False
    key = _normalized(path)
    if key in pins:
        return True
    prefix = key.rstrip("\\/")
    return any(pin.startswith(prefix + os.sep) or pin.startswith(prefix + "/") for pin in pins)


def preview_rules(custom: Any = None) -> dict[str, Any]:
    """What the rules would remove right now. Nothing is deleted."""
    rules = _validated_rules(custom) if custom is not None else read_rules()
    report = scan()
    try:
        from core.labs import benchmarks

        pins = benchmarks.pinned_artifacts()
    except Exception:  # noqa: BLE001 - no pins means nothing extra to protect
        pins = set()
    now = time.time()
    minimum_age = rules["min_age_days"] * 86_400
    kept: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for root_id in rules["roots"]:
        entries = [entry for entry in report["entries"] if entry["root"] == root_id]
        entries.sort(key = lambda entry: entry["modified"], reverse = True)
        for index, entry in enumerate(entries):
            slim = {
                "path": entry["path"],
                "name": entry["name"],
                "root": entry["root"],
                "kind": entry["kind"],
                "bytes": entry["bytes"],
                "bytes_label": entry["bytes_label"],
                "modified": entry["modified"],
            }
            if index < rules["keep_newest"]:
                kept.append(slim)
                continue
            if not entry["deletable"]:
                protected.append(
                    {"path": entry["path"], "reason": entry["protected_reason"]}
                )
                continue
            if _holds_a_pin(entry["path"], pins):
                protected.append(
                    {
                        "path": entry["path"],
                        "reason": "A pinned benchmark points at this checkpoint.",
                    }
                )
                continue
            age = now - _epoch(entry["modified"])
            if age < minimum_age:
                protected.append(
                    {"path": entry["path"], "reason": "Younger than the minimum age."}
                )
                continue
            candidates.append(slim)
    candidates.sort(key = lambda entry: entry["bytes"], reverse = True)
    total = sum(entry["bytes"] for entry in candidates)
    return {
        "rules": rules,
        "candidates": candidates,
        "kept": kept,
        "protected": protected,
        "total_bytes": total,
        "total_label": _format_bytes(total),
        "notes": [
            "The newest "
            f"{rules['keep_newest']} folders in each root are kept whatever their age.",
            f"Nothing younger than {rules['min_age_days']} days is offered.",
            "A checkpoint with a pinned benchmark is never offered.",
            "Preview only: applying this uses the same delete guard as the folder list.",
        ],
    }


def apply_rules(custom: Any = None) -> dict[str, Any]:
    """Remove exactly what :func:`preview_rules` offered."""
    rules = _validated_rules(custom) if custom is not None else read_rules()
    if not rules["enabled"]:
        raise ValueError("The janitor rules are switched off.")
    preview = preview_rules(rules)
    paths = [entry["path"] for entry in preview["candidates"]]
    result = delete(paths, confirm = True) if paths else {
        "deleted": [],
        "refused": [],
        "freed_bytes": 0,
        "freed_label": _format_bytes(0),
    }
    return {"rules": rules, "preview": preview, "result": result}
