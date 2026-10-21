"""SSN-BLIP2-v9: F_trans 进 dynamic_scalar 影响 s.

继承 v8. 唯一的架构改动:

  当前 v8:
      scalar_input = [text_embed, ref_embed, Fq_fused]    # 768 维
      s = dynamic_scalar(scalar_input)
      final_blend = s · text + (1 - s) · ref

  v9:
      scalar_input = [text_embed, ref_embed, Fq_fused, F_trans]    # 1024 维
      s = dynamic_scalar(scalar_input)
      final_blend = s · text + (1 - s) · ref

理由
----
v8 训完 γ ≈ 0.05, F_trans 通过末项 γ·F_trans 进入 final_query 几乎没影响.
原因是 F_trans 编码的"transition direction"跟 SSN blend `s·text+(1-s)·ref`
在几何上重叠 → 优化器选择把 γ 压小. 而这条 SSN blend 是 final_query 的
主导项 (系数 1.0).

v9 让 F_trans 通过 dynamic_scalar 调节 s, 间接控制 SSN blend 的混合比.
这样 F_trans 影响 final_query 主导项的方向, 不再被 γ=0.05 这条窄管道压死.

直觉
----
不同 action 类型本应该有不同的 s:
  * Remove: target 跟 ref 视觉极相似 → s 应偏小 (偏 ref)
  * Add: target 包含 ref 没有的物体 → s 应偏大 (偏 text)
  * Move/Attr: 介于两者之间

当前 dynamic_scalar 只看 text/ref/Fq_fused, 看不到清晰的 action 信号
(被整段 caption 的语义稀释). F_trans 是 ActionRouter 按 hard routing
分类输出的, 天然携带 action-explicit 信号. 喂给 dynamic_scalar 后,
它能学到 action-conditioned 的 s.

不动 v2-v8 / token_fusion / FFTBranch.
"""
