"""SSN-BLIP2-v6-stage: v6 + 阶段切换调度.

继承 v6, 在 forward 内根据当前 epoch 动态调整两个超参数:
  * mask_target_w  : box 正则的强度 (stage 2 加强)
  * lambda_kl_action: push-pull 损失权重 (stage 2 减弱)

阶段切换由训练入口写入 model._current_epoch buffer 触发, 模型自己
决定什么时候用哪套超参. 不改 baseline train_cirr_blip2.train.

设计原理
--------
复刻 v5 runB 的成功模式 (从零探索 30 epoch + fine-tune 20 epoch),
但合并到单阶段 30 epoch:

  Stage 1 (epoch [0, switch_epoch)): 标准超参, 让 mask head 充分探索
  Stage 2 (epoch [switch_epoch, end)): 加强 box / 减弱 KL, 精炼 mask

预期 epoch 25-29 的 5 个精炼 epoch 让 m_act_mean 收敛到 box 内部,
减少 KL push 带来的 mask 漂移, 拉高最终 val arithmetic.

v6 原模型不动. v5 / v4 / baseline 全部不动.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from SSN_v6.ssn_blip2_v6 import Model as V6Model  # noqa: E402


class Model(V6Model):
    """v6 + stage-aware schedule."""

    def __init__(self, *args, **kwargs):
        # v6-stage-only kwargs
        self._stage_switch_epoch = int(kwargs.pop("stage_switch_epoch", 25))
        self._stage2_mask_target_w = float(
            kwargs.pop("stage2_mask_target_w", 2.0)
        )
        self._stage2_lambda_kl_action = float(
            kwargs.pop("stage2_lambda_kl_action", 0.05)
        )
        # stage 1 默认值是 __init__ 阶段传进来的 mask_target_w/lambda_kl_action
        # 我们在 super().__init__ 后把它们记下来作为 stage 1 base
        super().__init__(*args, **kwargs)

        # Stage 1 baseline (super 设置了 self._mask_target_w / self.lambda_kl_action)
        self._stage1_mask_target_w = float(self._mask_target_w)
        self._stage1_lambda_kl_action = float(self.lambda_kl_action)

        # 当前 epoch buffer. 训练入口每 epoch 开头 fill_(epoch).
        # forward 读它决定走哪个 stage. -1 = stage 1 (默认)
        self.register_buffer(
            "_current_epoch",
            torch.tensor(-1, dtype=torch.long),
            persistent=False,
        )

    def _apply_stage_schedule(self) -> None:
        """Read self._current_epoch and apply stage-dependent hyperparams."""
        cur = int(self._current_epoch.item())
        if cur < 0 or cur < self._stage_switch_epoch:
            # Stage 1
            self._mask_target_w = self._stage1_mask_target_w
            self.lambda_kl_action = self._stage1_lambda_kl_action
        else:
            # Stage 2
            self._mask_target_w = self._stage2_mask_target_w
            self.lambda_kl_action = self._stage2_lambda_kl_action

    def forward(
        self,
        samples: Dict[str, Any],
        warmup: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Apply stage schedule before computing loss
        self._apply_stage_schedule()
        return super().forward(samples, warmup=warmup)
