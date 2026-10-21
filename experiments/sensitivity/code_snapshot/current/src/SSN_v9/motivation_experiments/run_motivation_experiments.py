"""Run SSN-v9 motivation experiments without modifying the model code.

The script evaluates three paper-motivation checks:

1. Reference-dependence / text-only shortcut:
   full composed retrieval vs text-only vs reference-only.

2. FOE object-size buckets:
   full model vs w/o FOE (no Ff / frequency branch), grouped by absolute
   ObjectLocalizer reference-object mask area thresholds.

3. Action perturbation:
   original graph action vs Add/Remove swap, with object, location, caption,
   reference image, and candidate set kept unchanged. The main reported
   causal subset contains only original Add/Remove samples.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_utils import (  # noqa: E402
    CIRRDataset,
    CustomCIRRDataset,
    PatternCIRDataset,
    squarepad_transform,
)
from model.action_transitions import ACTION_TO_CATEGORY, action_to_category  # noqa: E402
from train_cirr_blip2 import FeatureDataset, cir_collate_fn  # noqa: E402
from SSN_v9.ssn_blip2_v9 import Model as V9Model  # noqa: E402


def _custom_cirr_root() -> Path:
    packaged = Path(__file__).resolve().parents[3] / "data" / "CustomCIRR_graph"
    default = (packaged if packaged.exists()
               else Path("/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"))
    return Path(os.environ.get("CUSTOM_CIRR_ROOT", default))


CUSTOM_CIRR_ROOT = _custom_cirr_root()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "motivation_experiments.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def inject_object_core(inner_ds, split: str) -> None:
    if not hasattr(inner_ds, "triplets"):
        return
    src = CUSTOM_CIRR_ROOT / split / f"{split}_query_text_with_graph_object_core.json"
    if not src.exists():
        logging.warning("[object_core] not found: %s", src)
        return
    with open(src) as f:
        new_triplets = json.load(f)
    if len(new_triplets) != len(inner_ds.triplets):
        logging.warning(
            "[object_core] length mismatch: %d vs %d",
            len(new_triplets), len(inner_ds.triplets),
        )
        return
    inner_ds.triplets = new_triplets
    n = sum(1 for t in new_triplets if t.get("graph", {}).get("object_core"))
    logging.info("[object_core] injected %s (%d/%d)", split, n, len(new_triplets))


def get_gallery_names(dataset) -> List[str]:
    if hasattr(dataset, "image_paths"):
        return list(dataset.image_paths)
    if hasattr(dataset, "name_to_relpath"):
        return list(dataset.name_to_relpath.keys())
    if hasattr(dataset, "image_names"):
        return list(dataset.image_names)
    raise AttributeError(f"Cannot find gallery names on {type(dataset)}")


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        graph = {}
        entry = {}
        if hasattr(self.ds, "triplets") and idx < len(self.ds.triplets):
            entry = self.ds.triplets[idx]
            graph = entry.get("graph", {})
        return item[0], item[1], item[2], graph, item[3], idx


def load_dataset(args) -> FeatureDataset:
    preprocess = squarepad_transform(224)
    if args.dataset == "CustomCIRR":
        inner = CustomCIRRDataset(args.split, "relative", preprocess)
        inject_object_core(inner, args.split)
    elif args.dataset == "CIRR":
        inner = CIRRDataset(args.split, "relative", preprocess)
    elif args.dataset == "patterncir":
        inner = PatternCIRDataset(args.split, "relative", preprocess, with_graph=True)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    ds = FeatureDataset(inner)
    ds.load_image_features(args.feature_dir)
    return ds


def load_model(args, device: torch.device) -> V9Model:
    model = V9Model(
        selector_ckpt=args.selector_ckpt,
        max_txt_len=args.max_txt_len,
        top_k_fl=args.top_k_fl,
        fl_multi=args.fl_multi,
        ff_multi=args.ff_multi,
    )
    pretrained = "weight/blip2_pretrained.pth"
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
        logging.info("Loaded BLIP2 pretrained weights from %s", pretrained)
    else:
        logging.warning("Pretrained weight not found: %s", pretrained)

    state = torch.load(args.model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.info("Checkpoint missing keys: %d (first 8: %s)", len(missing), missing[:8])
    if unexpected:
        logging.info(
            "Checkpoint unexpected keys: %d (first 8: %s)",
            len(unexpected), unexpected[:8],
        )
    model = model.to(device).eval()
    model.set_ablation()
    model.aux_ramp.fill_(1.0)
    logging.info("Loaded checkpoint: %s", args.model_path)
    return model


@torch.no_grad()
def build_gallery(model: V9Model, ds: FeatureDataset, device: torch.device, batch_size: int):
    all_names = get_gallery_names(ds.dataset)
    chunks = []
    for i in tqdm(range(0, len(all_names), batch_size), desc="gallery", ncols=120):
        names = all_names[i : i + batch_size]
        raw = ds.get_image_features(names).to(device)
        raw_f = raw.to(next(model.parameters()).dtype)
        emb_ln = model.ln_vision(raw_f)
        chunks.append(model._qformer_image_only(emb_ln, model.query_tokens))
    gallery_h = torch.cat(chunks, dim=0)
    z_gallery = F.normalize(model.vision_proj(gallery_h), dim=-1)
    name_to_gidx = {n: i for i, n in enumerate(all_names)}
    return all_names, z_gallery, name_to_gidx


@torch.no_grad()
def text_only_query(model: V9Model, captions: Sequence[str]) -> torch.Tensor:
    tokens = model._tokenize(captions)
    out = model.Qformer.bert(
        tokens.input_ids,
        attention_mask=tokens.attention_mask,
        return_dict=True,
    )
    return F.normalize(model.text_proj(out.last_hidden_state[:, 0, :]), dim=-1)


@torch.no_grad()
def ref_only_query(model: V9Model, image_embeds_ln: torch.Tensor) -> torch.Tensor:
    img_tokens = model._qformer_image_only(image_embeds_ln, model.query_tokens)
    return F.normalize(model.vision_proj(img_tokens).mean(dim=1), dim=-1)


@torch.no_grad()
def full_query(
    model: V9Model,
    image_embeds_ln: torch.Tensor,
    captions: Sequence[str],
    graphs: Sequence[Any],
    no_ff: bool = False,
) -> torch.Tensor:
    prev = (model.ablate_no_trans, model.ablate_no_fl, model.ablate_no_ff,
            model.ablate_force_route)
    model.set_ablation(
        no_trans=prev[0],
        no_fl=prev[1],
        no_ff=no_ff,
        force_route=prev[3],
    )
    prev_graph_enabled = model.graph_enabled
    model.graph_enabled = True
    try:
        out = model._build_query(image_embeds_ln, list(captions), graphs)
        return out["final_query"]
    finally:
        model.graph_enabled = prev_graph_enabled
        model.set_ablation(
            no_trans=prev[0],
            no_fl=prev[1],
            no_ff=prev[2],
            force_route=prev[3],
        )


def graph_copy(g: Any) -> Dict[str, Any]:
    if isinstance(g, str):
        try:
            g = json.loads(g) if g.strip() else {}
        except json.JSONDecodeError:
            g = {}
    return copy.deepcopy(g if isinstance(g, dict) else {})


def replace_action_label(action: str) -> str:
    cat = action_to_category(action or "")
    if cat == "Add":
        return "remove"
    if cat == "Remove":
        return "add"
    if cat == "Move":
        return "remove"
    return "add"


def set_graph_action(g: Dict[str, Any], new_action: Any) -> None:
    """Change only the graph action signal.

    Keep object/location/caption/candidates unchanged, but update action-node
    labels that exactly match the previous action. Otherwise the old action
    node can leak back into object-node encoding after graph["action"] changes.
    """
    old_action = str(g.get("action", "") or "").strip().lower()
    new_action_s = str(new_action or "unknown")
    g["action"] = new_action_s
    for node in g.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        label = str(node.get("label", "") or "").strip().lower()
        if old_action and label == old_action:
            node["label"] = new_action_s


def replace_caption_action(caption: str, old_action: Any, new_action: Any) -> str:
    """Replace only the action phrase in the natural-language query.

    The edit is intentionally shallow: object/location/constraints remain as
    written, while the action verb/phrase is swapped. If the graph action text
    does not appear literally, fall back to the first known action phrase found
    in the caption.
    """
    text = str(caption)
    old = str(old_action or "").strip()
    new = str(new_action or "unknown").strip()

    def _case_like(src: str, repl: str) -> str:
        if src.isupper():
            return repl.upper()
        if src[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    candidates = []
    if old:
        candidates.append(old)
    candidates.extend(sorted(ACTION_TO_CATEGORY.keys(), key=len, reverse=True))

    seen = set()
    for cand in candidates:
        cand_norm = cand.strip().lower()
        if not cand_norm or cand_norm in seen:
            continue
        seen.add(cand_norm)
        pattern = re.compile(
            r"(?<![A-Za-z])" + re.escape(cand) + r"(?![A-Za-z])",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return text[:match.start()] + _case_like(match.group(0), new) + text[match.end():]
    return text


def add_remove_swap_action(action: Any) -> str | None:
    cat = action_to_category(action)
    if cat == "Add":
        return "remove"
    if cat == "Remove":
        return "add"
    return None


def perturb_graphs(graphs: Sequence[Any], mode: str, rng: random.Random) -> List[Dict[str, Any]]:
    out = [graph_copy(g) for g in graphs]
    if mode == "replace":
        for g in out:
            set_graph_action(g, replace_action_label(str(g.get("action", ""))))
        return out
    if mode == "add_remove_swap":
        for g in out:
            new_action = add_remove_swap_action(g.get("action", ""))
            if new_action is not None:
                set_graph_action(g, new_action)
        return out
    if mode == "shuffle":
        actions = [g.get("action", "unknown") for g in out]
        if len(actions) > 1:
            for _ in range(10):
                shuffled = actions[:]
                rng.shuffle(shuffled)
                if any(a != b for a, b in zip(actions, shuffled)):
                    actions = shuffled
                    break
        for g, a in zip(out, actions):
            set_graph_action(g, a)
        return out
    raise ValueError(f"Unknown action perturbation mode: {mode}")


def perturb_captions_for_actions(captions: Sequence[str],
                                 original_graphs: Sequence[Any],
                                 perturbed_graphs: Sequence[Any]) -> List[str]:
    out = []
    for cap, old_g, new_g in zip(captions, original_graphs, perturbed_graphs):
        old_action = graph_copy(old_g).get("action", "")
        new_action = graph_copy(new_g).get("action", "")
        out.append(replace_caption_action(str(cap), old_action, new_action))
    return out


def similarities(query: torch.Tensor, z_gallery: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,ntd->bnt", query, z_gallery).max(dim=-1).values


def metrics_from_sorted(
    sorted_names: np.ndarray,
    target_names: Sequence[str],
    group_members: Sequence[Sequence[str]],
    indices: Iterable[int] | None = None,
) -> Dict[str, float]:
    if indices is None:
        idx = np.arange(len(target_names))
    else:
        idx = np.array(list(indices), dtype=np.int64)
    if len(idx) == 0:
        return {
            "count": 0,
            "R@1": float("nan"), "R@5": float("nan"),
            "R@10": float("nan"), "R@50": float("nan"),
            "acc": float("nan"),
            "GS@1": float("nan"), "GS@2": float("nan"), "GS@3": float("nan"),
        }

    tgt = np.array(target_names, dtype=object)[idx]
    names = sorted_names[idx]
    labels = names == tgt[:, None]
    r1 = float(labels[:, :1].sum() / len(idx)) * 100
    r5 = float(labels[:, :5].sum() / len(idx)) * 100
    r10 = float(labels[:, :10].sum() / len(idx)) * 100
    r50 = float(labels[:, :50].sum() / len(idx)) * 100

    gs1 = gs2 = gs3 = 0
    valid = 0
    for row_i in idx:
        members = set(group_members[row_i])
        group_sorted = [n for n in sorted_names[row_i] if n in members]
        if not group_sorted:
            continue
        valid += 1
        tgt_name = target_names[row_i]
        if group_sorted[0] == tgt_name:
            gs1 += 1
        if tgt_name in group_sorted[:2]:
            gs2 += 1
        if tgt_name in group_sorted[:3]:
            gs3 += 1
    valid = max(valid, 1)
    return {
        "count": int(len(idx)),
        "R@1": r1, "R@5": r5, "R@10": r10, "R@50": r50,
        "acc": (r1 + r5 + r10 + r50) / 4,
        "GS@1": gs1 / valid * 100,
        "GS@2": gs2 / valid * 100,
        "GS@3": gs3 / valid * 100,
    }


def action_group_indices(graphs: Sequence[Any]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(graphs):
        gd = graph_copy(g)
        groups[action_to_category(gd.get("action", ""))].append(i)
    return dict(groups)


@torch.no_grad()
def evaluate_variant(
    tag: str,
    model: V9Model,
    ds: FeatureDataset,
    z_gallery: torch.Tensor,
    all_names: Sequence[str],
    name_to_gidx: Dict[str, int],
    device: torch.device,
    args,
    query_mode: str = "full",
    no_ff: bool = False,
    action_mode: str = "",
    perturb_text_action: bool = False,
) -> Dict[str, Any]:
    query_ds = QueryDataset(ds.dataset)
    loader = torch.utils.data.DataLoader(
        query_ds,
        batch_size=args.query_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cir_collate_fn,
    )
    rng = random.Random(args.seed)
    distances = []
    ref_names_all: List[str] = []
    tgt_names_all: List[str] = []
    group_members_all: List[List[str]] = []
    graphs_all: List[Any] = []
    idx_all: List[int] = []

    for batch in tqdm(loader, desc=tag, ncols=120):
        if batch is None:
            continue
        ref_names, tgt_names, captions, graphs, group_members, indices = batch
        eval_graphs = graphs
        eval_captions = captions
        if action_mode:
            eval_graphs = perturb_graphs(graphs, action_mode, rng)
            if perturb_text_action:
                eval_captions = perturb_captions_for_actions(captions, graphs, eval_graphs)

        raw = ds.get_image_features(ref_names).to(device)
        raw_f = raw.to(next(model.parameters()).dtype)
        emb_ln = model.ln_vision(raw_f)

        if query_mode == "full":
            q = full_query(model, emb_ln, eval_captions, eval_graphs, no_ff=no_ff)
        elif query_mode == "text":
            q = text_only_query(model, eval_captions)
        elif query_mode == "ref":
            q = ref_only_query(model, emb_ln)
        else:
            raise ValueError(f"Unknown query mode: {query_mode}")

        dist = similarities(q, z_gallery)
        distances.append(dist.cpu())
        ref_names_all.extend(list(ref_names))
        tgt_names_all.extend(list(tgt_names))
        graphs_all.extend(list(graphs))
        idx_all.extend([int(i) for i in indices])
        if isinstance(group_members[0], (list, tuple)):
            group_members_all.extend([list(g) for g in group_members])
        else:
            group_members_all.extend([[g] for g in group_members])

    distances_t = torch.cat(distances, dim=0)
    for i, rn in enumerate(ref_names_all):
        if rn in name_to_gidx:
            distances_t[i, name_to_gidx[rn]] = -1e10

    sorted_idx = torch.argsort(distances_t, dim=-1, descending=True).cpu().numpy()
    sorted_names = np.array(all_names, dtype=object)[sorted_idx]
    metrics = metrics_from_sorted(sorted_names, tgt_names_all, group_members_all)

    by_action = {}
    for action, ids in action_group_indices(graphs_all).items():
        by_action[action] = metrics_from_sorted(
            sorted_names, tgt_names_all, group_members_all, ids
        )

    return {
        "tag": tag,
        "metrics": metrics,
        "by_action": by_action,
        "sorted_names": sorted_names,
        "targets": tgt_names_all,
        "groups": group_members_all,
        "graphs": graphs_all,
        "indices": idx_all,
    }


@torch.no_grad()
def collect_object_ratios(
    model: V9Model,
    ds: FeatureDataset,
    device: torch.device,
    args,
) -> List[Dict[str, Any]]:
    query_ds = QueryDataset(ds.dataset)
    loader = torch.utils.data.DataLoader(
        query_ds,
        batch_size=args.query_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cir_collate_fn,
    )
    rows: List[Dict[str, Any]] = []
    prev_graph_enabled = model.graph_enabled
    model.graph_enabled = True
    try:
        for batch in tqdm(loader, desc="object-size", ncols=120):
            if batch is None:
                continue
            ref_names, _tgt_names, _captions, graphs, _group_members, indices = batch
            raw = ds.get_image_features(ref_names).to(device)
            raw_f = raw.to(next(model.parameters()).dtype)
            emb_ln = model.ln_vision(raw_f)
            ctx = model._build_action_inputs(emb_ln, graphs)
            mask = ctx["M_obj_ref"].detach().float().cpu()
            for local_i, idx in enumerate(indices):
                g = graph_copy(graphs[local_i])
                cat = action_to_category(g.get("action", ""))
                if args.object_size_area_mode == "soft":
                    ratio = float(mask[local_i].mean().item())
                else:
                    ratio = float(
                        (mask[local_i] >= args.object_size_mask_threshold)
                        .float().mean().item()
                    )
                rows.append({
                    "index": int(idx),
                    "ratio": ratio,
                    "action_category": cat,
                    "action": g.get("action", ""),
                    "object": g.get("object", ""),
                })
    finally:
        model.graph_enabled = prev_graph_enabled
    return rows


def assign_size_buckets(rows: List[Dict[str, Any]], args) -> Dict[str, List[int]]:
    valid = [r for r in rows if r["action_category"] != "Add"]
    if args.object_size_bucket_mode == "two":
        buckets = {"small": [], "non_small": []}
    elif args.object_size_bucket_mode == "extreme":
        buckets = {"small": [], "large": []}
    else:
        buckets = {"small": [], "medium": [], "large": []}

    for r in valid:
        ratio = r["ratio"]
        if args.object_size_bucket_mode == "two":
            if ratio < args.object_size_small_th:
                buckets["small"].append(r["index"])
            else:
                buckets["non_small"].append(r["index"])
        elif args.object_size_bucket_mode == "extreme":
            if ratio < args.object_size_small_th:
                buckets["small"].append(r["index"])
            elif ratio >= args.object_size_large_th:
                buckets["large"].append(r["index"])
            # Drop medium-range samples in extreme mode.
        else:
            if ratio < args.object_size_small_th:
                buckets["small"].append(r["index"])
            elif ratio < args.object_size_large_th:
                buckets["medium"].append(r["index"])
            else:
                buckets["large"].append(r["index"])
    return buckets


def summarize_size_experiment(full_res: Dict[str, Any], noff_res: Dict[str, Any],
                              rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    buckets_by_original_idx = assign_size_buckets(rows, args)
    pos_by_original_idx = {idx: pos for pos, idx in enumerate(full_res["indices"])}
    out = {}
    for bucket, original_indices in buckets_by_original_idx.items():
        positions = [
            pos_by_original_idx[i] for i in original_indices
            if i in pos_by_original_idx
        ]
        full_m = metrics_from_sorted(
            full_res["sorted_names"], full_res["targets"], full_res["groups"], positions
        )
        noff_m = metrics_from_sorted(
            noff_res["sorted_names"], noff_res["targets"], noff_res["groups"], positions
        )
        delta = {
            k: full_m[k] - noff_m[k]
            for k in ("R@1", "R@5", "R@10", "R@50", "acc", "GS@1", "GS@2", "GS@3")
            if not np.isnan(full_m[k]) and not np.isnan(noff_m[k])
        }
        out[bucket] = {"full": full_m, "wo_foe": noff_m, "delta": delta}
    return out


def summarize_add_remove_action(full_res: Dict[str, Any],
                                swap_res: Dict[str, Any]) -> Dict[str, Any]:
    positions = []
    for i, g in enumerate(full_res["graphs"]):
        cat = action_to_category(graph_copy(g).get("action", ""))
        if cat in ("Add", "Remove"):
            positions.append(i)

    full_m = metrics_from_sorted(
        full_res["sorted_names"], full_res["targets"], full_res["groups"], positions
    )
    swap_m = metrics_from_sorted(
        swap_res["sorted_names"], swap_res["targets"], swap_res["groups"], positions
    )
    delta = {
        k: swap_m[k] - full_m[k]
        for k in ("R@1", "R@5", "R@10", "R@50", "acc", "GS@1", "GS@2", "GS@3")
        if not np.isnan(full_m[k]) and not np.isnan(swap_m[k])
    }
    return {
        "subset": "original action category in {Add, Remove}",
        "positions": positions,
        "full": full_m,
        "add_remove_swap": swap_m,
        "delta_swap_minus_full": delta,
    }


def strip_large_arrays(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tag": res["tag"],
        "metrics": res["metrics"],
        "by_action": res["by_action"],
    }


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logging.info("Saved %s", path)


def print_metrics_table(title: str, results: Dict[str, Dict[str, Any]]) -> None:
    logging.info("")
    logging.info("==== %s ====", title)
    logging.info("%-18s %7s %7s %7s %7s %7s %7s %7s",
                 "tag", "R@1", "R@5", "R@10", "R@50", "GS@1", "GS@2", "GS@3")
    for tag, res in results.items():
        m = res["metrics"]
        logging.info(
            "%-18s %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f",
            tag, m["R@1"], m["R@5"], m["R@10"], m["R@50"],
            m["GS@1"], m["GS@2"], m["GS@3"],
        )


def print_subset_metrics_table(
    title: str,
    results: Dict[str, Dict[str, Any]],
    positions: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    logging.info("")
    logging.info("==== %s ====", title)
    logging.info("subset n=%d", len(positions))
    logging.info("%-24s %7s %7s %7s %7s %7s %7s %7s",
                 "tag", "R@1", "R@5", "R@10", "R@50", "GS@1", "GS@2", "GS@3")
    out = {}
    for tag, res in results.items():
        m = metrics_from_sorted(
            res["sorted_names"], res["targets"], res["groups"], positions
        )
        out[tag] = m
        logging.info(
            "%-24s %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f",
            tag, m["R@1"], m["R@5"], m["R@10"], m["R@50"],
            m["GS@1"], m["GS@2"], m["GS@3"],
        )
    return out


def parse_experiments(value: str) -> set[str]:
    exps = {x.strip().lower() for x in value.split(",") if x.strip()}
    allowed = {"shortcut", "size", "action"}
    bad = exps - allowed
    if bad:
        raise ValueError(f"Unknown experiments: {sorted(bad)}")
    return exps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="CustomCIRR",
                        choices=["CustomCIRR", "CIRR", "patterncir"])
    parser.add_argument("--split", default="val", choices=["val", "test", "test1"])
    parser.add_argument("--feature_dir", default="features/CustomCIRR")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--experiments", default="shortcut,size,action")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gallery_batch_size", type=int, default=256)
    parser.add_argument("--query_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--selector_ckpt", default="weight/selector_eva.pt")
    parser.add_argument("--max_txt_len", type=int, default=32)
    parser.add_argument("--top_k_fl", type=int, default=4)
    parser.add_argument("--fl_multi", type=lambda x: x.lower() in ("1", "true", "yes", "y"),
                        default=True)
    parser.add_argument("--ff_multi", type=lambda x: x.lower() in ("1", "true", "yes", "y"),
                        default=True)
    parser.add_argument("--object_size_area_mode", default="hard",
                        choices=["hard", "soft"],
                        help="hard: fraction of patches above mask threshold; "
                             "soft: mean mask response.")
    parser.add_argument("--object_size_bucket_mode", default="three",
                        choices=["three", "two", "extreme"],
                        help="three: small/medium/large; two: small/non_small; "
                             "extreme: small/large and drop medium-range samples.")
    parser.add_argument("--object_size_mask_threshold", type=float, default=0.5,
                        help="Patch is counted as object when M_obj >= this value "
                             "in hard area mode.")
    parser.add_argument("--object_size_small_th", type=float, default=0.10,
                        help="Object area ratio below this value is small.")
    parser.add_argument("--object_size_large_th", type=float, default=0.30,
                        help="Object area ratio at or above this value is large; "
                             "middle is medium.")
    args = parser.parse_args()

    set_seed(args.seed)
    exps = parse_experiments(args.experiments)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.model_path).parent / "motivation_experiments"
    )
    setup_logging(out_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logging.info("device=%s experiments=%s", device, sorted(exps))

    ds = load_dataset(args)
    model = load_model(args, device)
    all_names, z_gallery, name_to_gidx = build_gallery(
        model, ds, device, args.gallery_batch_size
    )

    cache: Dict[str, Dict[str, Any]] = {}

    def run_cached(tag: str, **kwargs):
        if tag not in cache:
            cache[tag] = evaluate_variant(
                tag, model, ds, z_gallery, all_names, name_to_gidx, device, args,
                **kwargs,
            )
        return cache[tag]

    if "shortcut" in exps:
        shortcut = {
            "full": run_cached("full", query_mode="full"),
            "text_only": run_cached("text_only", query_mode="text"),
            "ref_only": run_cached("ref_only", query_mode="ref"),
        }
        print_metrics_table("Reference dependence / text-only shortcut", shortcut)
        save_json(
            out_dir / f"shortcut_{args.dataset}_{args.split}.json",
            {k: strip_large_arrays(v) for k, v in shortcut.items()},
        )

    if "size" in exps:
        full_res = run_cached("full", query_mode="full")
        noff_res = run_cached("wo_foe_no_ff", query_mode="full", no_ff=True)
        rows = collect_object_ratios(model, ds, device, args)
        size_summary = summarize_size_experiment(full_res, noff_res, rows, args)
        logging.info("")
        logging.info("==== FOE object-size buckets ====")
        logging.info(
            "size rule: area_mode=%s, bucket_mode=%s, mask_th=%.3f, small<%.3f, large>=%.3f",
            args.object_size_area_mode,
            args.object_size_bucket_mode,
            args.object_size_mask_threshold,
            args.object_size_small_th,
            args.object_size_large_th,
        )
        for bucket, item in size_summary.items():
            fm, wm, dm = item["full"], item["wo_foe"], item["delta"]
            logging.info(
                "%-7s n=%4d  Full GS@1=%.2f R@1=%.2f | w/o FOE GS@1=%.2f R@1=%.2f | delta GS@1=%+.2f R@1=%+.2f",
                bucket, fm["count"], fm["GS@1"], fm["R@1"],
                wm["GS@1"], wm["R@1"], dm.get("GS@1", float("nan")),
                dm.get("R@1", float("nan")),
            )
        save_json(
            out_dir / f"size_buckets_{args.dataset}_{args.split}.json",
            {
                "overall": {
                    "full": strip_large_arrays(full_res),
                    "wo_foe": strip_large_arrays(noff_res),
                },
                "buckets": size_summary,
                "object_ratios": rows,
                "size_rule": {
                    "area_mode": args.object_size_area_mode,
                    "bucket_mode": args.object_size_bucket_mode,
                    "mask_threshold": args.object_size_mask_threshold,
                    "small": f"ratio < {args.object_size_small_th}",
                    "non_small": (
                        f"ratio >= {args.object_size_small_th}"
                        if args.object_size_bucket_mode == "two" else None
                    ),
                    "medium": (
                        f"{args.object_size_small_th} <= ratio < "
                        f"{args.object_size_large_th}"
                        if args.object_size_bucket_mode == "three" else None
                    ),
                    "large": (
                        f"ratio >= {args.object_size_large_th}"
                        if args.object_size_bucket_mode in ("three", "extreme")
                        else None
                    ),
                    "dropped_in_extreme": (
                        f"{args.object_size_small_th} <= ratio < "
                        f"{args.object_size_large_th}"
                        if args.object_size_bucket_mode == "extreme" else None
                    ),
                },
                "note": "Add actions are excluded from buckets because the edited object is not in the reference image.",
            },
        )

    if "action" in exps:
        full_res = run_cached("full", query_mode="full")
        swap_graph_res = run_cached(
            "action_add_remove_swap_graph_only",
            query_mode="full",
            action_mode="add_remove_swap",
        )
        swap_text_graph_res = run_cached(
            "action_add_remove_swap_text_graph",
            query_mode="full",
            action_mode="add_remove_swap",
            perturb_text_action=True,
        )
        add_remove_summary = summarize_add_remove_action(full_res, swap_text_graph_res)
        action_results = {
            "full": full_res,
            "swap_graph_only": swap_graph_res,
            "swap_text_graph": swap_text_graph_res,
            "action_shuffle": run_cached(
                "action_shuffle", query_mode="full", action_mode="shuffle"
            ),
        }
        subset_positions = add_remove_summary["positions"]
        subset_table = print_subset_metrics_table(
            "Action perturbation on changed Add/Remove subset",
            action_results,
            subset_positions,
        )
        logging.info("")
        logging.info("==== Add/Remove text+graph swap delta ====")
        fm = add_remove_summary["full"]
        sm = add_remove_summary["add_remove_swap"]
        dm = add_remove_summary["delta_swap_minus_full"]
        logging.info(
            "subset n=%d | full GS@1=%.2f R@1=%.2f | Add<->Remove text+graph swap GS@1=%.2f R@1=%.2f | delta GS@1=%+.2f R@1=%+.2f",
            fm["count"], fm["GS@1"], fm["R@1"],
            sm["GS@1"], sm["R@1"],
            dm.get("GS@1", float("nan")), dm.get("R@1", float("nan")),
        )
        save_json(
            out_dir / f"action_perturb_{args.dataset}_{args.split}.json",
            {
                "overall": {k: strip_large_arrays(v) for k, v in action_results.items()},
                "add_remove_subset": {
                    k: v for k, v in add_remove_summary.items()
                    if k != "positions"
                },
                "add_remove_subset_table": subset_table,
                "note": (
                    "swap_graph_only changes only graph action. "
                    "swap_text_graph changes both graph action and the action "
                    "phrase in the natural-language query. Add->remove and "
                    "Remove/Replace->add. Move-category samples are not "
                    "changed for this swap. The main printed table reports "
                    "only original Add/Remove samples, i.e. the changed subset."
                ),
            },
        )

    logging.info("Done. Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
