#!/usr/bin/env python3
"""Small stdlib-only helper used by the Windows launcher to preview a folder-derived trigger."""
from __future__ import annotations
import re
import sys
from pathlib import Path


def derive_trigger(folder_name: str) -> str:
    name = re.sub(r"(?i)(?:[_\-\s]+(?:source|dataset))$", "", folder_name).strip(" _-")
    aliases = []
    for part in re.split(r"__+", name):
        alias = re.sub(r"_+", " ", part)
        alias = re.sub(r"\s+", " ", alias).strip(" ,")
        if alias:
            aliases.append(alias.lower())
    return ", ".join(aliases)


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    trigger = derive_trigger(Path(sys.argv[1]).name)
    if not trigger:
        return 1
    print(trigger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
