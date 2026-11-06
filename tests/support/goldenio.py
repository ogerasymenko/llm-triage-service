"""A tiny YAML reader for samples/golden.yaml.

PyYAML would be a new dependency bought for one file read by one test, which the
spec's dependency rule does not justify. So this parses only the subset
golden.yaml uses. If that file grows features this cannot handle, that is the
moment to add the dependency rather than extend this.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SCALARS = {"null": None, "~": None, "true": True, "false": False}


def _scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if text.lower() in _SCALARS:
        return _SCALARS[text.lower()]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d*\.\d+", text):
        return float(text)
    return text


def _inline_list(raw: str) -> list[Any]:
    inner = raw.strip()[1:-1].strip()
    if not inner:
        return []
    parts, depth, current = [], 0, ""
    for char in inner:
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        current += char
    parts.append(current)
    return [_scalar(p) for p in parts]


def _value(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        return _inline_list(text)
    return _scalar(text)


class _Line:
    __slots__ = ("indent", "text")

    def __init__(self, indent: int, text: str) -> None:
        self.indent = indent
        self.text = text


def _lines(source: str) -> list[_Line]:
    out: list[_Line] = []
    for raw in source.splitlines():
        without_comment = re.sub(r"(?<!\S)#.*$", "", raw).rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip())
        out.append(_Line(indent, without_comment.strip()))
    return out


def _parse_block(lines: list[_Line], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return None, index

    if lines[index].text.startswith("- "):
        items: list[Any] = []
        while index < len(lines) and lines[index].indent == indent and lines[index].text.startswith("- "):
            body = lines[index].text[2:].strip()
            index += 1
            if ":" in body and not body.startswith("["):
                # A mapping whose first key shares the dash's line.
                key, _, rest = body.partition(":")
                mapping: dict[str, Any] = {}
                if rest.strip():
                    mapping[key.strip()] = _value(rest)
                else:
                    child, index = _parse_block(lines, index, lines[index].indent if index < len(lines) else indent + 2)
                    mapping[key.strip()] = child
                child_indent = indent + 2
                while index < len(lines) and lines[index].indent >= child_indent:
                    sub, index = _parse_mapping(lines, index, lines[index].indent)
                    mapping.update(sub)
                items.append(mapping)
            else:
                items.append(_value(body))
        return items, index

    return _parse_mapping_block(lines, index, indent)


def _parse_mapping(lines: list[_Line], index: int, indent: int) -> tuple[dict[str, Any], int]:
    mapping, index = _parse_mapping_block(lines, index, indent)
    return mapping, index


def _parse_mapping_block(lines: list[_Line], index: int, indent: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines) and lines[index].indent == indent and not lines[index].text.startswith("- "):
        key, _, rest = lines[index].text.partition(":")
        key = key.strip()
        rest = rest.strip()
        index += 1
        if rest in (">", "|", ">-", "|-"):
            chunks: list[str] = []
            while index < len(lines) and lines[index].indent > indent:
                chunks.append(lines[index].text)
                index += 1
            mapping[key] = " ".join(chunks).strip()
        elif rest:
            mapping[key] = _value(rest)
        else:
            if index < len(lines) and lines[index].indent > indent:
                child, index = _parse_block(lines, index, lines[index].indent)
                mapping[key] = child
            else:
                mapping[key] = None
    return mapping, index


def load_golden(path: str | Path) -> dict[str, Any]:
    lines = _lines(Path(path).read_text())
    if not lines:
        return {}
    data, _ = _parse_block(lines, 0, lines[0].indent)
    return data or {}
