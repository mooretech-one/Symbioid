"""Shared id helpers for Core classes."""

from uuid import uuid4


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid4().hex[:12]}"
