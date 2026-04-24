"""Generate an identity mapping YAML from a class-folder directory.

Useful when the external dataset's subfolder names already match the target
``data/<class>`` names; lets you skip hand-writing the mapping for
``ML/ingest_external.py``.

Usage:
    python ML/gen_identity_mapping.py <src_dir> <out_yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src = Path(argv[1])
    out = Path(argv[2])
    if not src.is_dir():
        print(f"src not a directory: {src}")
        return 1
    dirs = sorted(d.name for d in src.iterdir() if d.is_dir())
    print(f"{len(dirs)} classes found")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated identity map (gloss == folder)\n")
        for name in dirs:
            safe = name.replace('"', '\\"')
            f.write(f'"{safe}": "{safe}"\n')
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
