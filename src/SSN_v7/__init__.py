"""SSN-BLIP2-v7: drop M_loc, use spaCy core nouns for selector grounding.

What's different from v6_stage
------------------------------

1. **`graph.object` 由 spaCy 预处理成 `graph.object_core`（list of head nouns）**
   再喂给 ObjectLocalizer 做多 node grounding。"long boat-filled dock"
   → ["dock"], "road and trees" → ["road", "tree"]. M_obj 应该更聚焦。
   (`object_core` 字段由 scripts/preprocess_object_core.py 一次性生成.)

2. **完全不算 M_loc.** 可视化已经证明 ObjectLocalizer 对纯方位词
   ("left", "right", "upper") 几乎完全无能, M_loc 是噪声. v7 直接不算它,
   依赖 M_loc 的下游也全部改回 M_obj-based:

       - SemanticInjector: `α · obj_proto.broadcast()` (全图均匀注入)
       - AddTransitionV7: 4-input `[obj_proto, obj_feat, loc_feat, action_feat]`
         (去掉 region; Add 类 M_obj 是 uniform = Fg, 加进去等于冗余)
       - MoveTransitionV7: 4-input `[region(M_obj), obj_feat, loc_feat, action_feat]`
         (去掉 region_dst, 退回 v5 风格)

3. **Remove / Attr Transition 不变** (它们本来就只用 M_obj).

阶段调度 / token_fusion 4-token / l_trans_align 等 v6_stage 的设计 **全部沿用**.
"""
