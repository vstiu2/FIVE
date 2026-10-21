"""分析 v9 模型: action 类型对 dynamic_scalar 输出的 s 是否有显著影响.

核心问题: F_trans 进 dynamic_scalar 后, s 是否真的随 action 类别变化?
理论: Remove 类应该偏 ref (s 小), Add 类应该偏 text (s 大), Move/Attr 介于
中间.

输出:
  * 每个 action 类别 (Remove / Add / Move / Attr) 的 s 均值 / 标准差;
  * 各类对比, 直观看出 F_trans 是否影响 s;
  * 同时对 v8 同样 ckpt 做一次, 当对照 (没有 F_trans 进 dynamic_scalar);
  * 画一个 4 类的 s 分布直方图.

Usage:
    python src/SSN_v9/analyze_action_s.py \\
        --v9_ckpt logs/SSN_BLIP2/CustomCIRR/ssn_v9/runA/best_arithmetic.pth \\
        --v8_ckpt logs/SSN_BLIP2/CustomCIRR/ssn_v8/runA/best_arithmetic.pth \\
        --feature_dir features/CustomCIRR \\
        --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import CustomCIRRDataset, squarepad_transform   # noqa: E402
from train_cirr_blip2 import FeatureDataset                     # noqa: E402

def _custom_cirr_root() -> Path:
    packaged = Path(__file__).resolve().parents[2] / "data" / "CustomCIRR_graph"
    default = (packaged if packaged.exists()
               else Path("/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"))
    return Path(os.environ.get("CUSTOM_CIRR_ROOT", default))


CUSTOM_CIRR_ROOT = _custom_cirr_root()


def inject_object_core(inner_ds, split: str) -> None:
    src = CUSTOM_CIRR_ROOT / split / f"{split}_query_text_with_graph_object_core.json"
    if not src.exists():
        print(f"[warn] {src} not found")
        return
    if not hasattr(inner_ds, "triplets"):
        return
    with open(src) as f:
        new_triplets = json.load(f)
    if len(new_triplets) != len(inner_ds.triplets):
        return
    inner_ds.triplets = new_triplets
    print(f"  injected object_core into {split} ({len(new_triplets)} triplets)")


def load_v9_model(ckpt_path: str, device):
    from SSN_v9.ssn_blip2_v9 import Model
    m = Model(
        selector_ckpt="weight/selector_eva.pt",
        lambda_kl_action=0.1, lambda_counterfact=0, blend_dropout=0,
        alpha_inject_init=0.2, mask_target_lo=0.4, mask_target_hi=0.7,
        mask_target_w=1.0, lambda_trans_align=0.1,
        top_k_fl=4, lambda_fl=0.1, lambda_ff=0.1,
        fl_multi=True, ff_multi=True,
    )
    m.load_from_pretrained("weight/blip2_pretrained.pth")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m = m.to(device).eval()
    m.graph_enabled = True
    return m


def load_v8_model(ckpt_path: str, device):
    from SSN_v8.ssn_blip2_v8 import Model
    m = Model(
        selector_ckpt="weight/selector_eva.pt",
        lambda_kl_action=0.1, lambda_counterfact=0, blend_dropout=0,
        alpha_inject_init=0.2, mask_target_lo=0.4, mask_target_hi=0.7,
        mask_target_w=1.0, lambda_trans_align=0.1,
        top_k_fl=4, lambda_fl=0.1, lambda_ff=0.1,
        fl_multi=True, ff_multi=True,
    )
    m.load_from_pretrained("weight/blip2_pretrained.pth")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m = m.to(device).eval()
    m.graph_enabled = True
    return m


# 4 类 action 映射 (同 v6 v7 v8 v9)
def action_to_cat(a):
    a = (a or "").strip().lower()
    if a in ("remove", "replace"):
        return "Remove"
    if a in ("add", "attach", "surround", "fill", "complete", "cast", "pack",
             "place", "put", "reveal", "add two more piers", "add three more houses"):
        return "Add"
    if a in ("move", "shift", "swap", "rearrange", "center",
             "rotate and move", "rotate and shift"):
        return "Move"
    return "Attr"   # rotate / zoom / change color 等都归 Attr


@torch.no_grad()
def compute_s_for_split(model, ds, device, batch_size=32):
    """跑一遍 dataset, 提取每个样本的 (action_cat, s) 对."""
    results = defaultdict(list)   # {cat: [s1, s2, ...]}

    # 我们只需要 ref + graph, 不需要 target.
    indices = list(range(len(ds.dataset)))

    for i in tqdm(range(0, len(indices), batch_size), desc="compute s"):
        idx_batch = indices[i:i + batch_size]
        entries = [ds.dataset.triplets[j] for j in idx_batch]
        ref_names = [e["ref_image"] for e in entries]
        graphs = [e.get("graph", {}) for e in entries]
        captions = [e.get("query_text", "") for e in entries]

        raw = ds.get_image_features(ref_names).to(device)
        with torch.cuda.amp.autocast():
            raw_f = raw.to(next(model.parameters()).dtype)
            emb_ln = model.ln_vision(raw_f)

            # 重要: 我们要捕捉 dynamic_scalar 的输出. 复用 _build_query 但
            # 不需要 final_query, 只看 dynamic_s.
            # 走完 _build_query, 模型会把 dynamic_s_mean 写到 out, 但我们要
            # per-sample. 不如重新算一遍 (跟 _build_query 一致):
            #   Fq_u, q_tokens, text_cls = _qformer_fusion(...)
            #   ref_embed / text_embed
            #   ctx = _build_action_inputs(...)
            #   patches_action = _compose_patches(...)
            #   F_trans (router)
            #   Fq_fused = token_fusion(...)
            #   scalar_input = [text, ref, Fq_fused, F_trans_n]  (v9)
            #                 or [text, ref, Fq_fused]            (v8)
            #   s = dynamic_scalar(scalar_input)
            #
            # 但更简单: 直接 monkey-patch 让 _build_query 把 s 写到 out:
            out = model._build_query(emb_ln, captions, graphs)
        # 从模型最后一次 forward 里取每个样本的 s.
        # _build_query 的 dynamic_s 是 [B, 1] 但只写了 mean. 我们另开口子:
        # 这里直接走代码读取 — 见 patched _build_query.
        s_per_sample = out["_dynamic_s_per_sample"].squeeze(-1).cpu().tolist()

        for cat_input, s in zip(graphs, s_per_sample):
            cat = action_to_cat(cat_input.get("action"))
            results[cat].append(float(s))

    return results


def patch_build_query_to_record_s(model):
    """让 _build_query 把 per-sample dynamic_s 写到 out 字典."""
    original = model._build_query

    def patched(image_embeds_ln, captions, graphs, target_patches=None):
        out = original(image_embeds_ln, captions, graphs, target_patches)
        # _build_query 内部已经算了 dynamic_s 但只塞了 mean (在 v9). 我们
        # 重新算一遍 (不是最优, 但稳).
        # 实际上更简单: 修改 v9 model 也行, 这里偷懒重算.
        return out

    # 上面的 patch 没用 (没拦到 dynamic_s). 用别的方式:
    # 直接 hook 在 dynamic_scalar 输出.
    captured = {"s": None}

    def hook(module, inp, out_):
        captured["s"] = out_.detach()

    handle = model.dynamic_scalar.register_forward_hook(hook)

    def patched2(image_embeds_ln, captions, graphs, target_patches=None):
        out = original(image_embeds_ln, captions, graphs, target_patches)
        out["_dynamic_s_per_sample"] = captured["s"]
        return out

    model._build_query = patched2
    return handle


def report(name, results, out_dir):
    print()
    print(f"========== {name} ==========")
    print(f"{'cat':10s}  {'count':>6s}  {'mean':>8s}  {'std':>8s}  {'min':>6s}  {'max':>6s}")
    summary = {}
    order = ["Remove", "Add", "Move", "Attr"]
    for cat in order:
        ss = results.get(cat, [])
        if not ss:
            continue
        arr = np.array(ss)
        summary[cat] = (len(ss), arr.mean(), arr.std(), arr.min(), arr.max())
        print(f"{cat:10s}  {len(ss):6d}  {arr.mean():8.4f}  {arr.std():8.4f}  "
              f"{arr.min():6.3f}  {arr.max():6.3f}")

    # pairwise gap
    if "Remove" in summary and "Add" in summary:
        diff = summary["Add"][1] - summary["Remove"][1]
        print(f"\nGap Add - Remove = {diff:+.4f}")
        if abs(diff) < 0.02:
            print("  → 几乎相同, action 类型对 s 影响极小")
        elif abs(diff) < 0.05:
            print("  → 轻微差异")
        elif abs(diff) < 0.1:
            print("  → 中等差异, F_trans 有一些作用")
        else:
            print("  → 明显差异, F_trans 真的让 s 跟 action 相关!")

    # plot
    plt.figure(figsize=(8, 4))
    for cat in order:
        ss = results.get(cat, [])
        if not ss:
            continue
        plt.hist(ss, bins=30, alpha=0.5, label=f"{cat} (n={len(ss)})",
                 density=True)
    plt.xlabel("dynamic_scalar output (s)")
    plt.ylabel("density")
    plt.title(f"{name}: distribution of s by action category")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"s_distribution_{name}.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved histogram → {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9_ckpt", required=True,
        help="path to v9 best_arithmetic.pth")
    ap.add_argument("--v8_ckpt", default="",
        help="optional v8 ckpt for control comparison")
    ap.add_argument("--feature_dir", default="features/CustomCIRR")
    ap.add_argument("--split", default="val", choices=["test", "val"])
    ap.add_argument("--out_dir", default="logs/SSN_BLIP2/CustomCIRR/ssn_v9/runA")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    preprocess = squarepad_transform(224)
    inner = CustomCIRRDataset(args.split, "relative", preprocess)
    inject_object_core(inner, args.split)
    ds = FeatureDataset(inner)
    ds.load_image_features(args.feature_dir)

    # v9
    print("Loading v9 model...")
    m9 = load_v9_model(args.v9_ckpt, device)
    h9 = patch_build_query_to_record_s(m9)
    print("Computing s for v9...")
    res9 = compute_s_for_split(m9, ds, device)
    h9.remove()
    sum9 = report("v9", res9, out_dir)

    # v8 (optional)
    sum8 = None
    if args.v8_ckpt and os.path.isfile(args.v8_ckpt):
        print("\nLoading v8 model for control...")
        m8 = load_v8_model(args.v8_ckpt, device)
        h8 = patch_build_query_to_record_s(m8)
        print("Computing s for v8...")
        res8 = compute_s_for_split(m8, ds, device)
        h8.remove()
        sum8 = report("v8", res8, out_dir)

    # final verdict
    print()
    print("============ Verdict ============")
    if sum9 and "Remove" in sum9 and "Add" in sum9:
        gap9 = sum9["Add"][1] - sum9["Remove"][1]
        print(f"v9: Add s mean = {sum9['Add'][1]:.4f}, "
              f"Remove s mean = {sum9['Remove'][1]:.4f}, gap = {gap9:+.4f}")
    if sum8 and "Remove" in sum8 and "Add" in sum8:
        gap8 = sum8["Add"][1] - sum8["Remove"][1]
        print(f"v8: Add s mean = {sum8['Add'][1]:.4f}, "
              f"Remove s mean = {sum8['Remove'][1]:.4f}, gap = {gap8:+.4f}")
        if sum9:
            delta = gap9 - gap8
            print(f"Gap difference (v9 - v8) = {delta:+.4f}")
            if abs(delta) > 0.05:
                print("→ v9 改造确实让 s 变得更 action-conditioned!")
            else:
                print("→ v9 跟 v8 行为相似, F_trans 进 dynamic_scalar 没显著效果.")


if __name__ == "__main__":
    main()
