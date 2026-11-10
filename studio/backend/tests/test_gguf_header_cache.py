# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The GGUF header cache: one parse per unchanged file, and no stale answers.

The header of a GGUF is asked the same three questions -- architecture, quant,
context length -- every time a model is listed, validated or loaded. The file does not
change between those questions, so the answer is memoized against the file's own
identity (``path``, size, mtime).
"""

from __future__ import annotations

import os
import struct

import pytest

from utils import gguf_archs


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _make_gguf(
    architecture: str = "llama",
    file_type: int = 15,
    context_length: int = 4096,
) -> bytes:
    """A minimal GGUF header: magic, version, no tensors, three metadata entries."""
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 3)
    body += _string("general.architecture") + struct.pack("<I", 8) + _string(architecture)
    body += _string("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type)
    body += (
        _string(f"{architecture}.context_length")
        + struct.pack("<I", 4)
        + struct.pack("<I", context_length)
    )
    return body


@pytest.fixture(autouse = True)
def _clean_cache():
    """Each test starts from an empty cache, so the bound is never crossed by accident."""
    gguf_archs.invalidate_header_cache()
    yield
    gguf_archs.invalidate_header_cache()


def _write_gguf(tmp_path, name: str = "model.gguf", **kwargs) -> str:
    path = tmp_path / name
    path.write_bytes(_make_gguf(**kwargs))
    return str(path)


def test_two_calls_parse_once(tmp_path, monkeypatch):
    """The second call for an unchanged file is answered from the cache."""
    path = _write_gguf(tmp_path)

    calls = []
    real_parse = gguf_archs.parse_gguf_header

    def counting_parse(target: str):
        calls.append(target)
        return real_parse(target)

    monkeypatch.setattr(gguf_archs, "parse_gguf_header", counting_parse)

    first = gguf_archs.read_gguf_header(path)
    second = gguf_archs.read_gguf_header(path)

    assert len(calls) == 1, f"the header must be parsed once, not {len(calls)} times"
    assert first == second
    assert first["architecture"] == "llama"
    assert first["quant"] == "Q4_K_M"
    assert first["context_length"] == 4096


def test_a_rewritten_file_is_parsed_again(tmp_path, monkeypatch):
    """A file whose bytes changed is a different file, and gets a fresh answer."""
    path = _write_gguf(tmp_path)

    calls = []
    real_parse = gguf_archs.parse_gguf_header

    def counting_parse(target: str):
        calls.append(target)
        return real_parse(target)

    monkeypatch.setattr(gguf_archs, "parse_gguf_header", counting_parse)

    before = gguf_archs.read_gguf_header(path)
    assert before["context_length"] == 4096
    assert len(calls) == 1

    # A rewrite that changes the size AND the mtime: the key is (path, size, mtime).
    with open(path, "wb") as handle:
        handle.write(_make_gguf(context_length = 8192))
    # Windows filesystems have a coarse mtime; make the change visible either way.
    stat = os.stat(path)
    os.utime(path, ns = (stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    after = gguf_archs.read_gguf_header(path)

    assert len(calls) == 2, "a rewritten file must be parsed again"
    assert after["context_length"] == 8192


def test_invalidate_forces_a_reparse(tmp_path, monkeypatch):
    """``invalidate()`` drops the memo, so the next call reads the file again."""
    path = _write_gguf(tmp_path)

    calls = []
    real_parse = gguf_archs.parse_gguf_header

    def counting_parse(target: str):
        calls.append(target)
        return real_parse(target)

    monkeypatch.setattr(gguf_archs, "parse_gguf_header", counting_parse)

    gguf_archs.read_gguf_header(path)
    assert len(calls) == 1

    gguf_archs.invalidate_header_cache()
    gguf_archs.read_gguf_header(path)
    assert len(calls) == 2, "invalidate() must force the next call to read the file"

    # The per-path form drops only that path.
    other = _write_gguf(tmp_path, name = "other.gguf")
    gguf_archs.read_gguf_header(other)
    assert len(calls) == 3

    gguf_archs.invalidate_header_cache(path)
    gguf_archs.read_gguf_header(path)
    assert len(calls) == 4
    gguf_archs.read_gguf_header(other)
    assert len(calls) == 4, "invalidating one path must not drop another"


def test_the_bound_evicts(tmp_path, monkeypatch):
    """The store is bounded: the oldest entry is dropped once the bound is passed."""
    cache = gguf_archs.GGUFHeaderCache(max_entries = 2)

    calls = []

    def counting_parse(target: str):
        calls.append(target)
        return {"architecture": "llama", "quant": "Q4_K_M", "context_length": 4096}

    paths = []
    for index in range(3):
        path = _write_gguf(tmp_path, name = f"model{index}.gguf")
        paths.append(path)
        cache.get(path, counting_parse)

    assert len(calls) == 3
    assert len(cache) == 2, "the store must not grow past its bound"

    # The first path was evicted (least recently used), so it is parsed again.
    cache.get(paths[0], counting_parse)
    assert len(calls) == 4, "the evicted entry must be parsed again"

    # And the most recently used entry is still there.
    cache.get(paths[2], counting_parse)
    assert len(calls) == 4, "the surviving entry must still be cached"


def test_the_default_bound_is_128():
    assert gguf_archs.HEADER_CACHE.max_entries == 128


def test_a_bulky_tokenizer_array_is_skipped_not_read(tmp_path):
    """A vocabulary is tens of thousands of entries; the walk must skip over it.

    The fields wanted here are in the general and architecture-scoped blocks, which a
    GGUF writes before the tokenizer, so the array is stepped over rather than read.
    """
    vocabulary = [f"token{i}\x00" for i in range(5000)]
    array = struct.pack("<IQ", 8, len(vocabulary))  # element type 8 (string), count
    for token in vocabulary:
        array += _string(token)

    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 4)
    body += _string("general.architecture") + struct.pack("<I", 8) + _string("qwen3")
    body += _string("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", 15)
    body += _string("qwen3.context_length") + struct.pack("<I", 4) + struct.pack("<I", 32768)
    body += _string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + array

    path = tmp_path / "vocab.gguf"
    path.write_bytes(body)

    header = gguf_archs.parse_gguf_header(str(path))

    assert header["architecture"] == "qwen3"
    assert header["quant"] == "Q4_K_M"
    assert header["context_length"] == 32768


def test_a_missing_file_raises_and_is_not_cached(tmp_path):
    """A missing file raises what it always raised, and nothing about it is stored."""
    missing = str(tmp_path / "nope.gguf")

    with pytest.raises(FileNotFoundError):
        gguf_archs.read_gguf_header(missing)

    assert len(gguf_archs.HEADER_CACHE) == 0, "a failed parse must not be cached"

    # And it raises again: the absence was not remembered as an answer.
    with pytest.raises(FileNotFoundError):
        gguf_archs.read_gguf_header(missing)


def test_a_corrupt_header_raises_and_is_not_cached(tmp_path):
    """A file that is not a GGUF raises, and the failure is not memoized."""
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"not a gguf at all")

    with pytest.raises(ValueError):
        gguf_archs.read_gguf_header(str(path))

    assert len(gguf_archs.HEADER_CACHE) == 0, "a failed parse must not be cached"


def test_returned_dicts_are_independent(tmp_path):
    """A caller that mutates the dict it got back cannot poison the next caller."""
    path = _write_gguf(tmp_path)

    first = gguf_archs.read_gguf_header(path)
    first["architecture"] = "MUTATED"
    first["injected"] = True

    second = gguf_archs.read_gguf_header(path)

    assert second["architecture"] == "llama"
    assert "injected" not in second
    assert first is not second


def test_the_cache_is_keyed_on_size_and_mtime(tmp_path):
    """The key is the file's identity: same path, different bytes, different entry."""
    path = _write_gguf(tmp_path)
    gguf_archs.read_gguf_header(path)
    assert len(gguf_archs.HEADER_CACHE) == 1

    with open(path, "wb") as handle:
        handle.write(_make_gguf(context_length = 2048))
    stat = os.stat(path)
    os.utime(path, ns = (stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    gguf_archs.read_gguf_header(path)
    assert len(gguf_archs.HEADER_CACHE) == 2, "the rewritten file is a second entry"


def test_a_missing_file_raises_what_opening_it_raises(tmp_path):
    """The cached path must not change the error a caller sees."""
    missing = str(tmp_path / "absent.gguf")

    with pytest.raises(FileNotFoundError) as cached:
        gguf_archs.read_gguf_header(missing)
    with pytest.raises(FileNotFoundError) as direct:
        open(missing, "rb")

    assert type(cached.value) is type(direct.value)
    assert cached.value.errno == direct.value.errno
    assert cached.value.filename == direct.value.filename


def test_concurrent_reads_get_one_answer(tmp_path):
    """Threads asking for the same file at the same time all get the same answer."""
    import threading

    path = _write_gguf(tmp_path)

    calls = []
    lock = threading.Lock()

    def counting_parse(target: str):
        with lock:
            calls.append(target)
        return gguf_archs.parse_gguf_header(target)

    cache = gguf_archs.GGUFHeaderCache(max_entries = 8)
    results = []
    threads = [
        threading.Thread(target = lambda: results.append(cache.get(path, counting_parse)))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(result == results[0] for result in results)
    assert results[0]["architecture"] == "llama"
    # Concurrent callers may each start a parse before the first one lands; the cache
    # guarantees one ANSWER and one stored entry, and the bound is never crossed.
    assert len(cache) == 1
