# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Short-lived, single-purpose link tokens: one capability, one scope, one expiry.

A session bearer in a query string outlives the click that made it: it lands in shell history, a
reverse proxy's access log and the ``Referer`` of anything the destination links to, and it carries
every right its subject has. A link token carries one right instead -- ``verify`` hands back claims,
never a session -- and stops working at ``exp``.

Format, two base64url segments joined by a dot::

    <base64url(json claims)>.<base64url(HMAC-SHA256(secret, domain || segment))>

The claims are ``scope`` (which endpoint the link opens), ``sub`` (which subject, or which signed
account target, it was minted for), ``iat`` and ``exp`` (unix seconds) and ``nonce`` (so two links
minted in one second still differ). A caller may add its own keys, such as the id the link names;
the reserved five are refused as payload keys so a payload can never rewrite them.

Not a JWT, deliberately:

* No ``alg`` field to negotiate. A JWT verifier that honours the header invites the classic
  ``alg: none`` and RS256-to-HS256 confusions; here the only algorithm is HMAC-SHA256 with a secret
  the caller passes in, so there is nothing to choose and nothing to downgrade.
* No dependency. The Studio already refuses to import a package for a few lines of ``hmac`` and
  ``json``, and a self-contained module is one an audit can read in full.
* No ``iss``/``aud``/``kid`` machinery to keep consistent with the account JWT. A link is not a
  credential and must never be accepted anywhere that takes one: it is signed with a secret of its
  own and its ``scope`` is checked on every ``verify``.

The secret is the caller's: this module never reads one from config, so a link can only be verified
by the process that minted it, and restarting a process invalidates its links (see ``secrets`` at
the call site). Comparison is constant time, and a token is refused whole rather than repaired.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Mapping, Optional

__all__ = ["ScopedLinkError", "mint", "verify", "TOKEN_MAX_LENGTH"]

# A link is a URL parameter, so it is bounded like one: long enough for a model id in the payload,
# short enough that a megabyte of base64 is never decoded.
TOKEN_MAX_LENGTH = 4096
# Domain separation: the same secret used for anything else cannot produce a signature that
# verifies here, and vice versa.
_DOMAIN = b"unsloth-scoped-link\0"
_RESERVED = ("scope", "sub", "exp", "iat", "nonce")
_DIGEST_BYTES = hashlib.sha256().digest_size


class ScopedLinkError(Exception):
    """A link that is malformed, tampered with, expired, or minted for another scope."""


def _now() -> float:
    """The clock, behind a name so a test can move it and a caller can reason about ``exp``."""
    return time.time()


def _secret_bytes(secret: Any) -> bytes:
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    if not isinstance(secret, (bytes, bytearray, memoryview)):
        raise ScopedLinkError("secret must be bytes or str")
    key = bytes(secret)
    if not key:
        raise ScopedLinkError("secret must not be empty")
    return key


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    """Strict: unlike a plain ``urlsafe_b64decode`` this rejects a stray character instead of
    dropping it, so one signature has exactly one spelling."""
    if not text.isascii():
        raise ScopedLinkError("link is not ASCII")
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), altchars = b"-_", validate = True)
    except (binascii.Error, ValueError) as exc:
        raise ScopedLinkError("link is not base64url") from exc


def _signature(secret: bytes, segment: str) -> bytes:
    return hmac.new(secret, _DOMAIN + segment.encode("ascii"), hashlib.sha256).digest()


def mint(
    secret: Any,
    scope: str,
    subject: str,
    payload: Optional[Mapping[str, Any]] = None,
    ttl_s: int = 300,
) -> str:
    """A link token for ``subject`` in ``scope``, good for ``ttl_s`` seconds.

    ``subject`` is opaque to this module: a subject name, a signed account target, anything the
    verifier will recognise. ``payload`` adds caller claims (``item``, ``path``, ...) alongside the
    reserved ones; the returned token is signed, so a verifier sees them only as minted.
    """
    key = _secret_bytes(secret)
    if not isinstance(scope, str) or not scope:
        raise ScopedLinkError("scope must be a non-empty string")
    if not isinstance(subject, str) or not subject:
        raise ScopedLinkError("subject must be a non-empty string")
    if not isinstance(ttl_s, int) or isinstance(ttl_s, bool) or ttl_s <= 0:
        raise ScopedLinkError("ttl_s must be a positive int")
    claims: dict[str, Any] = {}
    if payload:
        taken = [name for name in _RESERVED if name in payload]
        if taken:
            raise ScopedLinkError(f"payload may not set {', '.join(taken)}")
        claims.update(payload)
    issued = int(_now())
    claims.update(
        {
            "scope": scope,
            "sub": subject,
            "iat": issued,
            "exp": issued + ttl_s,
            "nonce": secrets.token_urlsafe(16),
        }
    )
    try:
        body = json.dumps(claims, separators = (",", ":"), sort_keys = True, allow_nan = False)
    except (TypeError, ValueError) as exc:
        raise ScopedLinkError("payload is not JSON-serialisable") from exc
    segment = _b64encode(body.encode("utf-8"))
    token = f"{segment}.{_b64encode(_signature(key, segment))}"
    if len(token) > TOKEN_MAX_LENGTH:
        raise ScopedLinkError(f"link is longer than {TOKEN_MAX_LENGTH} characters")
    return token


def verify(secret: Any, token: str, scope: str) -> dict[str, Any]:
    """The claims of a link minted for ``scope``, or ``ScopedLinkError``.

    The signature is checked before the claims are read, so nothing an attacker wrote is looked at
    until this process has proved it wrote it.
    """
    key = _secret_bytes(secret)
    if not isinstance(token, str) or not token:
        raise ScopedLinkError("link must be a non-empty string")
    if len(token) > TOKEN_MAX_LENGTH:
        raise ScopedLinkError(f"link is longer than {TOKEN_MAX_LENGTH} characters")
    segment, _, signature = token.partition(".")
    if not segment or not signature or "." in signature:
        raise ScopedLinkError("link is malformed")
    given = _b64decode(signature)
    # Bytes, equal length: compare_digest raises TypeError on a non-ASCII str, and a short
    # signature is refused rather than being compared against a prefix.
    if len(given) != _DIGEST_BYTES or not hmac.compare_digest(given, _signature(key, segment)):
        raise ScopedLinkError("link signature does not match")
    try:
        claims = json.loads(_b64decode(segment).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScopedLinkError("link claims are not JSON") from exc
    if not isinstance(claims, dict):
        raise ScopedLinkError("link claims are not an object")
    if claims.get("scope") != scope:
        raise ScopedLinkError("link was minted for another scope")
    for name in ("sub", "nonce"):
        if not isinstance(claims.get(name), str) or not claims[name]:
            raise ScopedLinkError(f"link claim {name} is missing")
    for name in ("iat", "exp"):
        if not isinstance(claims.get(name), int) or isinstance(claims[name], bool):
            raise ScopedLinkError(f"link claim {name} is not a whole number of seconds")
    # ``exp`` is exclusive: the second it names is not a second the link works in.
    if claims["exp"] <= int(_now()):
        raise ScopedLinkError("link has expired")
    if claims["exp"] <= claims["iat"]:
        raise ScopedLinkError("link expires before it was issued")
    return claims
