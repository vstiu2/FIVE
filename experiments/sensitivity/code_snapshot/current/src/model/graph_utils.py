"""Graph utilities: node text conversion and BERT-based node feature encoding."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import BertTokenizer


def graph_to_node_texts(graph: Any, empty_text: str = "graph node: unknown") -> List[str]:
    if isinstance(graph, str):
        try:
            graph = json.loads(graph) if graph.strip() else None
        except json.JSONDecodeError:
            graph = None
    if not isinstance(graph, dict):
        return [empty_text]

    node_texts: List[str] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        label = node.get("label") or node.get("text") or node.get("name") or "unknown"
        node_texts.append(f"graph node: {label}")

    # Append top-level semantic fields as extra pseudo-nodes when available
    for field in ("action", "object", "location", "result_location"):
        value = graph.get(field)
        if value and value != "unknown":
            node_texts.append(f"graph {field}: {value}")

    return node_texts or [empty_text]


def encode_graph_node_features(
    tokenizer: BertTokenizer,
    bert_model: torch.nn.Module,
    graphs: Sequence[Any],
    device: torch.device,
    max_txt_len: int = 32,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Encode graph nodes with the BERT tokenizer + Q-Former text encoder.

    Returns:
        node_features: [B, M, hidden_size]  (padded)
        node_mask:     [B, M]  bool
    """
    all_node_texts = [graph_to_node_texts(g) for g in graphs]
    max_nodes = max(len(t) for t in all_node_texts)

    flat_texts: List[str] = []
    node_mask = torch.zeros(len(graphs), max_nodes, dtype=torch.bool, device=device)
    for i, texts in enumerate(all_node_texts):
        node_mask[i, : len(texts)] = True
        flat_texts.extend(texts)
        flat_texts.extend([texts[-1]] * (max_nodes - len(texts)))  # pad by repeating last

    tokens = tokenizer(
        flat_texts,
        padding="max_length",
        truncation=True,
        max_length=max_txt_len,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = bert_model(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
            return_dict=True,
        )
    # Use CLS token (position 0) as node feature
    node_feats = out.last_hidden_state[:, 0, :]  # [B*M, H]
    if normalize:
        node_feats = F.normalize(node_feats, dim=-1)
    node_feats = node_feats.view(len(graphs), max_nodes, -1)  # [B, M, H]
    return {"node_features": node_feats, "node_mask": node_mask}
