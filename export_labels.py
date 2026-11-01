"""Export local human labels to a portable labels.json backup."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

from labels_store import connect_labels, export_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("labels.db"))
    parser.add_argument("--output", type=Path, default=Path("labels.json"))
    args = parser.parse_args()
    with closing(connect_labels(args.db)) as connection:
        count = export_labels(connection, args.output)
    print(f"Exported {count:,} labels to {args.output}")


if __name__ == "__main__":
    main()
