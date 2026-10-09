# WACQNet protocol v1.0.0 撤回记录

- 原冻结哈希：`be3b8ecd2282a3e015d3e4d94a17274451dcecffcaec653a2f61bb376dadf61d`
- 撤回发生在任何 WACQNet 训练之前。
- 原因：v1.0.0 写明了训练随机种子 `20260927`，但没有把历史 Level-B 图像生成种子 `42` 单独登记。若直接实现，可能造成重建图像与既有 frozen-HPR `mu_fixed` 行不一致。
- 修正：v1.0.1 增加 `level_b.generation_seed = 42`，训练随机种子仍为 `training.training_seed = 20260927`。
- 科学影响：无模型结果产生、无候选比较、无 Test 访问，因此该修正不受结果驱动。

