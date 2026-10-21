"""SSN-BLIP2-v6: F_trans 全面修复版。

What's new vs v5
----------------
v5 已经把 mask 做到了能给整体带 +1 个点。但 F_trans 这条 action 分支始终
没起作用——γ 训完只到 0.05，hard routing 让 Add/Move 数据不足，AddTransition
的 region 等于全图 mean 而不带"新物体"信息，Move 的 region 是源位置而不
是目标位置。v6 一次性修复:

(1) ActionRouter 由 3 类拆成 **4 类**:
       Remove / Add / Move / Attr (= Rotate + Resize + Recolor)
    "Attr" 把"物体位置不动、但属性变化"的子动作合并为一类，避免它们
    被混在 Move 里互相干扰，同时保留足够的样本量。

(2) AddTransition 改用 **M_loc + obj_proto**（来自 SemanticInjector）作
    为视觉输入，让 F_trans 真正看到"新物体应该在哪里 + 它的视觉概念"，
    而不是 ref 全图的 mean。

(3) MoveTransition 输入加上 **dst region**（M_loc 加权的 patches），让
    F_trans 表达"从 M_obj 位置 → M_loc 位置"的迁移，而不是只看源位置。

(4) trans_mlp 末层加 **LayerNorm**，让 F_trans 输出 magnitude 不再失控。

(5) F_trans 同时走 **两个出口**:
       a) 加入 token_fusion 作为第 4 个 K/V token   (借 β·Fq_fused 宽通道)
       b) 保留 γ·F_trans 这一项                     (向后兼容 v5 的设计)
    这样 attention 可以学到何时该看 F_trans，γ 这条窄通道作为补充。

(6) 新增辅助损失 **l_trans_align**:
       cosine_align(F_trans, target_embed - ref_embed.detach())
    给 F_trans 一个直接对齐 "ref→target transition direction" 的监督，
    不再完全依赖 InfoNCE 通过 γ=0.05 反传的微弱梯度。

v4 / v5 / token_fusion 原有代码不修改 (token_fusion 的 4-th token 参数
是可选的，默认行为不变).
"""
