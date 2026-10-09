# WACQNet protocol v1.0.1 撤回记录

- 原冻结哈希：`82a9b4239650cadeaf11604fc5f04c6f1253d6e55989b3b55a247c1d8cd1b5fb`
- 撤回发生在任何 WACQNet 参数训练之前；当时只进行Level-B输入缓存构建。
- 原因：v1.0.1 固定了网络拟合 repetition 0/25，但没有显式冻结LOBO内部 `q_hat` 的独立样本。
- 修正：v1.0.2 固定模型拟合 repetition 0/25，内部conformal校准使用互不重叠的 repetition 1/13/26/38。
- 科学影响：无训练结果、无候选比较、无 Test 访问；该修正用于消除潜在的训练—校准行泄漏。

