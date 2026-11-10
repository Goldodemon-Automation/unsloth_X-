# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The speech-architecture verdict, as a leaf module with no package baggage.

It lives here rather than beside the rest of the GGUF metadata helpers because
``utils.models.__init__`` eagerly imports ``model_config``, which imports ``yaml``.
``core.inference.llama_cpp`` needs this verdict at import time, and reaching it through
``utils.models`` made the whole models package -- and PyYAML with it -- a hard import
dependency of the chat backend. That took the repo's own Source lint job red, where
``tests/studio/load_freeze/test_load_orchestrator.py`` imports the backend without PyYAML
installed.

Every caller imports from here, so there is exactly ONE definition.

It also reads the GGUF HEADER -- architecture, quant and context length -- with a bounded
cache, for the same leaf-module reason: the reader is stdlib-only (``struct``), so nothing
here can pull ``utils.models`` (and with it PyYAML) into a caller that only wants the
header fields.
"""

from __future__ import annotations

import os
import struct
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

# ``general.architecture`` values no Unsloth runtime can decode (llama.cpp has no CSM decoder).
# Published CSM GGUFs disagree on spelling, so all four on the Hub are listed. Named once so the
# chat gate, the listing classifier and the media preflight cannot drift apart.
SPEECH_GGUF_ARCHS = frozenset({"llama-csm", "csm", "csm-tts", "mimi"})

# The Mimi vocoder in ggml-org/sesame-csm-1b-GGUF puts a whole SENTENCE in general.architecture
# rather than an identifier. Matched on the flag, not the full string, so a reword still lands.
_VOCODER_MARKERS = ("--model-vocoder", "cannot be used as llm")


def is_speech_gguf_architecture(architecture: Optional[str]) -> bool:
    """Whether ``general.architecture`` names something only a TTS runtime can decode.

    Case- and space-insensitive, like every other architecture comparison here. ``None`` and the
    empty string are NOT speech: a GGUF declaring no architecture is unknown, and every caller
    fails open on unknown."""
    if not architecture:
        return False
    normalized = architecture.strip().lower()
    if normalized in SPEECH_GGUF_ARCHS:
        return True
    return any(marker in normalized for marker in _VOCODER_MARKERS)


# ``general.architecture`` of every GGUF audio.cpp writes. Only audiocpp_server reads these (speech,
# music, transcription); llama.cpp has no such architecture.
AUDIO_CPP_GGUF_ARCHITECTURE = "audiocpp"


def is_audio_cpp_gguf_architecture(architecture: Optional[str]) -> bool:
    """Whether ``general.architecture`` names a GGUF only audio.cpp runs. Case- and space-insensitive."""
    return bool(architecture) and architecture.strip().lower() == AUDIO_CPP_GGUF_ARCHITECTURE


# ---------------------------------------------------------------------------
# The GGUF header: architecture, quant and context length, read once per file.
# ---------------------------------------------------------------------------
#
# A model is listed, validated and loaded from the same GGUF, and every one of those
# asks the header the same three questions. The header does not change while the file
# does not, so the answer is memoized against the file's own identity.
#
# The read is byte-bounded: the walker skips over the values it does not want (a
# tokenizer array is megabytes) instead of loading them, so the win from caching is
# avoiding SYSCALLS -- the open, the reads, the seeks -- rather than saving memory.
#
# The reader is stdlib-only on purpose. Reaching the header through ``utils.models``
# would run that package's ``__init__``, and with it PyYAML, into every importer of
# this leaf module; see the module docstring.

_GGUF_MAGIC = 0x46554747  # b"GGUF", little endian
_GGUF_HEADER_BYTES = 24  # magic, version, tensor count, kv count

# Value type -> fixed width in bytes. Strings and arrays are variable and skipped.
_FIXED_VTYPE_SIZES: Dict[int, int] = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 1,  # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
_VTYPE_STRING = 8
_VTYPE_ARRAY = 9

# Sanity bounds, so a corrupt header cannot ask for a gigabyte of memory.
_MAX_KEY_BYTES = 1 << 20
_MAX_STRING_BYTES = 1 << 22
_MAX_ARRAY_ELEMENTS = 1 << 30

# llama.cpp's file-type enum (``general.file_type``), which is how a GGUF names its
# quantization. Unknown values (newer llama.cpp releases add them) read as None
# rather than as a guess.
_FILE_TYPE_NAMES: Dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q5_0",
    8: "Q5_1",
    9: "Q8_0",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
}


def _read_exact(handle, count: int) -> bytes:
    """Read exactly *count* bytes or raise: a short read means a truncated header."""
    data = handle.read(count)
    if len(data) != count:
        raise ValueError("truncated GGUF header")
    return data


def _read_u32(handle) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_string(handle) -> str:
    length = _read_u64(handle)
    if length > _MAX_STRING_BYTES:
        raise ValueError("GGUF string length out of range")
    return _read_exact(handle, length).decode("utf-8", "replace")


def _skip_value(handle, vtype: int) -> None:
    """Advance past one GGUF value without reading its payload.

    This is what keeps the header read byte-bounded: a tokenizer array is skipped by
    seeking over it, so its bytes are never read at all.
    """
    if vtype == _VTYPE_STRING:
        length = _read_u64(handle)
        if length > _MAX_STRING_BYTES:
            raise ValueError("GGUF string length out of range")
        handle.seek(length, 1)
        return
    if vtype == _VTYPE_ARRAY:
        element_type = _read_u32(handle)
        count = _read_u64(handle)
        if count > _MAX_ARRAY_ELEMENTS:
            raise ValueError("GGUF array length out of range")
        if element_type == _VTYPE_STRING:
            # A string array stores each element behind its own length, so the
            # lengths must be read one by one; the payloads are still seeked over.
            for _ in range(count):
                length = _read_u64(handle)
                if length > _MAX_STRING_BYTES:
                    raise ValueError("GGUF string length out of range")
                handle.seek(length, 1)
            return
        size = _FIXED_VTYPE_SIZES.get(element_type)
        if size is None:
            raise ValueError(f"unknown GGUF array element type {element_type}")
        handle.seek(size * count, 1)
        return
    size = _FIXED_VTYPE_SIZES.get(vtype)
    if size is None:
        raise ValueError(f"unknown GGUF value type {vtype}")
    handle.seek(size, 1)


def parse_gguf_header(path: str) -> Dict[str, Any]:
    """Read the header of the GGUF at *path*.

    Returns ``{"architecture": str | None, "quant": str | None, "context_length": int | None}``.
    ``None`` means the header did not carry the field, not that the file was unreadable.

    Raises ``OSError`` when the file cannot be opened (a missing file raises
    ``FileNotFoundError``, exactly as opening it directly would) and ``ValueError``
    when the bytes are not a GGUF header at all.
    """
    architecture: Optional[str] = None
    file_type: Optional[int] = None
    contexts: Dict[str, int] = {}

    with open(path, "rb") as handle:
        magic, _version, _tensor_count, kv_count = struct.unpack(
            "<IIQQ", _read_exact(handle, _GGUF_HEADER_BYTES)
        )
        if magic != _GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")

        for _ in range(kv_count):
            key = _read_string(handle)
            vtype = _read_u32(handle)
            if vtype == _VTYPE_STRING and key == "general.architecture":
                architecture = _read_string(handle)
            elif vtype == 4 and key == "general.file_type":
                file_type = _read_u32(handle)
            elif vtype in (4, 10) and key.endswith(".context_length"):
                # The architecture lives in the key's own prefix ("llama.context_length"),
                # so the value can be filed under it before general.architecture is read.
                prefix = key[: -len(".context_length")]
                width = 4 if vtype == 4 else 8
                contexts[prefix] = struct.unpack(
                    "<I" if width == 4 else "<Q", _read_exact(handle, width)
                )[0]
            else:
                _skip_value(handle, vtype)
            if architecture is not None and file_type is not None and architecture in contexts:
                break

    context_length = contexts.get(architecture) if architecture is not None else None
    if context_length is None and len(contexts) == 1:
        # One candidate and no architecture to match it against: it is the only one there is.
        context_length = next(iter(contexts.values()))

    return {
        "architecture": architecture,
        "quant": _FILE_TYPE_NAMES.get(file_type) if file_type is not None else None,
        "context_length": context_length,
    }


def _cache_path(path: str) -> str:
    """One spelling per file, so two spellings of one path are one cache entry."""
    return os.path.abspath(os.fspath(path))


class GGUFHeaderCache:
    """A bounded, thread-safe memo of parsed GGUF headers.

    The key is ``(path, st_size, st_mtime_ns)``: the file's own identity. A rewritten
    file changes its size or its mtime, so it lands on a different key and is parsed
    again; a file that did not change keeps its answer.

    A parse that RAISES is never stored. A missing file, an unreadable file and a
    corrupt header are all answers that can change the moment the file is fixed, and
    memoizing one of them would hand the next caller a stale verdict. The error
    propagates instead, so the caller sees exactly what it would have seen without
    the cache.

    The store is bounded and evicts least-recently-used first: an unbounded memo over
    every model ever listed is the same defect as reading every header forever.
    """

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: "OrderedDict[Tuple[str, int, int], Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str, parse: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
        """The parsed header of *path*, parsed at most once per unchanged file.

        The returned dict is a fresh copy every time, so a caller that mutates what
        it got back cannot poison the next caller.
        """
        try:
            stat = os.stat(path)
        except OSError:
            # Unstattable: let the parse raise the error the caller expects (a missing
            # file raises FileNotFoundError from the open), and cache nothing.
            return dict(parse(path))

        key = (_cache_path(path), int(stat.st_size), int(stat.st_mtime_ns))
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                # Re-insert: the OrderedDict order is the eviction order, so a hit is
                # also a "this one is still in use".
                self._entries[key] = self._entries.pop(key)
                return dict(cached)

        # Parsed outside the lock: a slow disk read must not block other paths.
        parsed = parse(path)

        with self._lock:
            self._entries[key] = dict(parsed)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last = False)
        return dict(parsed)

    def invalidate(self, path: Optional[str] = None) -> None:
        """Forget every entry for *path*, or everything when *path* is None.

        The path is matched on its identity, so the entry goes whether it was stored
        under a path that has since been rewritten or not.
        """
        with self._lock:
            if path is None:
                self._entries.clear()
                return
            target = _cache_path(path)
            for key in [key for key in self._entries if key[0] == target]:
                del self._entries[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# The one cache the backend shares. Bound 128: enough for the models on a busy
# machine, small enough that the memo cannot grow without end.
HEADER_CACHE = GGUFHeaderCache(max_entries = 128)


def read_gguf_header(path: str) -> Dict[str, Any]:
    """The GGUF header fields of *path*, read once per unchanged file.

    See :func:`parse_gguf_header` for the fields and for what is raised.
    """
    return HEADER_CACHE.get(path, parse_gguf_header)


def invalidate_header_cache(path: Optional[str] = None) -> None:
    """Forget the cached header of *path*, or all of them when *path* is None."""
    HEADER_CACHE.invalidate(path)
