"""SSN-BLIP2-v8: Fl/Ff 多 token + Fl/Ff 直接监督.

继承 v7. 主要改动:

1. **Fl 多 token (top-K=4)**: 当前 v2-v7 的 Fl 是 M_obj 加权 mean → [B, 256].
   v8 取 M_obj 加权下的 top-K patches, 每个 patch 经 local_proj 投影成
   独立 [B, 256] token, 拼成 [B, K, 256] 进 token_fusion.

2. **Ff 多 token (4 频段)**: FFTBranch 内部本来分 4 个频段, v2-v7 用 softmax
   权重 mean 成 [B, 256]. v8 让 FFT 返回 [B, 4, 256] (一个频段一个 token),
   各频段独立进 token_fusion.

3. **token_fusion K/V token 总数**: Fg(1) + Fl(4) + Ff(4) + F_trans(1) = 10.
   (token_fusion.py 已改成 dual-mode, 支持单 token 和多 token 输入.)

4. **Fl/Ff 直接监督**:
       fl_query = normalize(fl_retrieval_head(mean_pool(Fl_multi)))
       ff_query = normalize(ff_retrieval_head(mean_pool(Ff_multi)))
   各加一个 InfoNCE: max-sim against z_tgt_tokens.

v2-v7 不动. token_fusion / reference_fft 改成向后兼容 (默认行为不变).
"""
