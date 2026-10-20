"""Verify the two trained checkpoints without unpickling their contents."""
import argparse
import hashlib
from pathlib import Path

EXPECTED = {
    "FIVE_PatternCIRRS.pth": "4c5ccfb6a802b76ac293a6c24f12fbab9080c0610917217be26f02775cb4f67a",
    "FIVE_PatternCIR.pth": "0da1d080b231de43bc47607cfdf24cef5e54e199ecfa97f827aef2184034bec1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    failures = []
    for name, expected in EXPECTED.items():
        path = args.assets_root / "weights" / name
        if not path.is_file():
            failures.append(f"Missing: {path}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != 2842846488 or digest.hexdigest() != expected:
            failures.append(f"Checksum/size mismatch: {path}")
        else:
            print(f"OK {name}: {digest.hexdigest()}")
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
