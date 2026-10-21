"""SSN-BLIP2-v10: per-class mask + top-K visual aggregation, drop F_trans MLP and Fg.

继承 v8. 核心改动:

1. 新增 ``PerClassActionBranch`` (src/SSN_v10/per_class_action.py):
   * 共享 backbone (~985K 参数): 所有 4 类样本都过, 学通用视觉/动作理解.
   * 末层 per-class (4 × 513 = 2K 参数): 每类一个 Linear head, 学
     "我这类该把高分给哪种 patch".
   * 输出 M_cls [B, N] (mask 分数) + F_action [B, K=8, embed_dim]
     (top-K 高分 patch 的视觉聚合, 不经过抽象 MLP).

2. _build_query 改动:
   * **删 Fg**: 全图 mean pool 跟 Q-Former Fq_u 冗余, 删掉.
   * **删 F_trans MLP 路径**: 不再调 v8 继承的 ActionRouterV7, 不再有
     γ·F_trans 末项, 不再把 F_trans 进 token_fusion.
   * **新 F_action 进 token_fusion**: F_action [B, 8, 256] 作为多 K/V
     token, 跟 Fl/Ff 并列.
   * token_fusion 总 K/V: Fl(4) + Ff(4) + F_action(8) = 16.

3. 保留 v8 全部其他设计:
   * 共享 M_act (ActionMaskHead) 给 Fl/Ff 用 (前端 gate);
   * SemanticInjector 全图均匀注入;
   * 4 类 hard routing (Remove/Add/Move/Attr);
   * v6_stage 的 25+10 stage 调度;
   * v7 的 object_core / drop M_loc;
   * v8 的 Fl/Ff 多 token + 直接监督 (lambda_fl/lambda_ff).

v2-v9 / token_fusion (dual-mode) / FFT branch (per-band 模式) 全部不动.

Hyper-parameters
----------------
* top_k_action = 8 (F_action 多 token 数量)
* per_class_backbone_dim = 512
* 沿用 v8 的 stage 调度 / object_core / Fl-Ff 监督等
"""
