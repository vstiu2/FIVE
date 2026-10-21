"""Run the archived motivation checks with the released FIVE resource layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run import ROOT, build_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["PatternCIRRS", "PatternCIR"], default="PatternCIRRS")
    parser.add_argument("--assets-root", default=str(ROOT))
    parser.add_argument("--data-root")
    parser.add_argument("--images-root")
    parser.add_argument("--feature-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--experiments", default="shortcut,size,action")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-threshold", type=float, choices=[0.3, 0.4], default=0.4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.mode = "eval"
    _, env, required = build_command(args, [])
    assets = Path(args.assets_root).expanduser().resolve()
    internal = "CustomCIRR" if args.dataset == "PatternCIRRS" else "patterncir"
    split = "test" if args.dataset == "PatternCIRRS" else "val"
    checkpoint = Path(args.checkpoint or assets / "weights" / f"FIVE_{args.dataset}.pth").resolve()
    features = Path(args.feature_dir or ROOT / "features" / internal).resolve()
    output = Path(args.output_dir or ROOT / "outputs/motivation" / args.dataset /
                  f"size_{args.size_threshold:g}").resolve()
    required.extend([assets / "weight/selector_eva.pt", assets / "weight/blip2_pretrained.pth"])
    if args.dataset == "PatternCIRRS":
        required.append(Path(env["CUSTOM_CIRR_ROOT"]) / split /
                        f"{split}_query_text_with_graph_object_core.json")
    command = [sys.executable,
               str(ROOT / "src/SSN_v9/motivation_experiments/run_motivation_experiments.py"),
               "--dataset", internal, "--split", split, "--model_path", str(checkpoint),
               "--feature_dir", str(features), "--out_dir", str(output),
               "--selector_ckpt", env["FIVE_SELECTOR_CKPT"], "--gpu", args.gpu,
               "--experiments", args.experiments, "--seed", str(args.seed),
               "--object_size_area_mode", "hard", "--object_size_bucket_mode", "two",
               "--object_size_mask_threshold", "0.5",
               "--object_size_small_th", str(args.size_threshold)]
    if args.dry_run:
        print(json.dumps({"command": command, "cwd": str(ROOT),
                          "resources": [str(p) for p in required]}, indent=2))
        return
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        parser.exit(2, "Missing resources:\n" + "\n".join(missing) + "\n")
    if args.check:
        print(f"Resource check passed: {len(required)} files. No model was loaded.")
        return
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
