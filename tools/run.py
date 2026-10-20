"""Launch the recorded FIVE experiments with explicit local resource paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BOOL_VALUES = {"f_trans_via_fusion", "f_trans_via_gamma", "fl_multi", "ff_multi"}


def build_command(args, extra):
    internal_dataset = "CustomCIRR" if args.dataset == "PatternCIRRS" else "patterncir"
    assets = Path(args.assets_root).expanduser().resolve()
    features = Path(args.feature_dir or ROOT / "features" / internal_dataset).resolve()
    data = Path(args.data_root or ROOT / "data" / (
        "CustomCIRR_graph" if args.dataset == "PatternCIRRS" else "PatternCIR_graph"
    )).resolve()
    images = Path(args.images_root or ROOT / "data" / "PatternNet").resolve()
    env = os.environ.copy()
    env.update({
        "CUSTOM_CIRR_ROOT": str(data),
        "PATTERNCIR_GRAPH_ROOT": str(data),
        "PATTERNNET_ROOT": str(images),
        "BERT_ROOT": str(assets / "bert-base-uncased"),
        "FIVE_EVA_CKPT": str(assets / "weight" / "eva_vit_g.pth"),
        "FIVE_BLIP2_PRETRAINED": str(assets / "weight" / "blip2_pretrained.pth"),
        "FIVE_SELECTOR_CKPT": str(assets / "weight" / "selector_eva.pt"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    required = [assets / "weight" / "eva_vit_g.pth",
                assets / "bert-base-uncased" / "config.json",
                assets / "bert-base-uncased" / "vocab.txt",
                assets / "bert-base-uncased" / "pytorch_model.bin"]
    if args.mode == "train":
        config = json.loads((ROOT / "configs" / f"{args.dataset}.json").read_text())
        config.update(dataset=internal_dataset, feature_dir=str(features), gpu=args.gpu,
                      selector_ckpt=env["FIVE_SELECTOR_CKPT"],
                      exp_name=args.exp_name or f"ssn_v9/reproduce_{args.dataset}")
        command = [sys.executable, str(ROOT / "src/SSN_v9/train_ssn_v9.py")]
        for key, value in config.items():
            if key in BOOL_VALUES:
                command.extend([f"--{key}", str(value).lower()])
            elif isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
            elif value != "" and value is not None:
                command.extend([f"--{key}", str(value)])
        splits = ["train", "val"]
        required.extend([assets / "weight/blip2_pretrained.pth",
                         assets / "weight/selector_eva.pt"])
    else:
        split = "test" if args.dataset == "PatternCIRRS" else "val"
        checkpoint = Path(args.checkpoint or assets / "weights" /
                          f"FIVE_{args.dataset}.pth").resolve()
        output = Path(args.output_dir or ROOT / "outputs" / args.dataset).resolve()
        command = [sys.executable, str(ROOT / "src/SSN_v9/test_ssn_v9.py"),
                   "--dataset", internal_dataset, "--split", split,
                   "--feature_dir", str(features), "--gpu", args.gpu,
                   "--model_path", str(checkpoint), "--output_dir", str(output)]
        required.append(checkpoint)
        splits = [split]
    for split in splits:
        required.extend([features / f"{split}_images.pt",
                         features / f"{split}_name_to_idx.json"])
        if args.dataset == "PatternCIRRS":
            required.extend([data / split / f"{split}_query_text_with_graph.json",
                             data / split / f"{split}_split_image.json"])
        else:
            required.append(data / split / f"cap.rc2.{split}.json")
    if args.dataset == "PatternCIR":
        required.extend(images / "image_splits" / f"split.rc2.{s}.json"
                        for s in ("train", "val", "test1"))
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    return command, env, required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--dataset", choices=["PatternCIRRS", "PatternCIR"], required=True)
    parser.add_argument("--assets-root", default=str(ROOT),
                        help="Directory containing weight/, weights/ and bert-base-uncased/.")
    parser.add_argument("--data-root", help="Graph annotation root; see README for schemas.")
    parser.add_argument("--images-root", help="PatternNet root containing image_splits/.")
    parser.add_argument("--feature-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--exp-name")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Print command without checking files.")
    parser.add_argument("--check", action="store_true", help="Check resource files without loading a model.")
    args, extra = parser.parse_known_args()
    command, env, required = build_command(args, extra)
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
