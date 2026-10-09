# BTI-WACQNet 数学协议（训练前冻结版 v1.0.0）

## 1. 研究对象与数据边界

本阶段只使用 36 个 **Development 背景组**：原始 Development 13 组、已经明确降级为开发数据的历史 Calibration 12 组，以及已经明确降级为开发数据的 v2 Calibration 11 组。当前 v3 的 19 个 Calibration 与 12 个 Confirmatory Test，以及所有历史 Confirmatory Test，均禁止读取。

外层统计单位是 `background_id`，realization 只是在一个背景内重复生成的受控 Level-B 样本，不被当作独立背景。

## 2. 可部署输入

对样本 (x)，冻结 HPR 给出锚点预测 (mu_0(x))。WACQNet 只使用观测时可得到的量：实际输入的 XCO2、(u/v) 风场、有效像元掩膜、源位置与冻结 operational plume template、以及 (mu_0(x))。真值 (Q)、APE、可靠性标签、背景 ID/类型和任何 Test 表现不得进入网络输入。

空间输入被确定性转换为 16×16 的六通道张量：稳健中心化 XCO2、稳健尺度化 XCO2、有效掩膜、冻结 plume template、顺风坐标与横风坐标。标量输入为 (log(1+mu_0))、(log(1+U_{in})) 和观测有效率。

## 3. 模型

卷积分支记为 (F=E_\theta(x)\)。冻结 template (T\in[0,1]) 与有效掩膜 (M) 定义：

\[
z_p=\frac{\sum TMF}{\sum TM+\epsilon},\qquad
z_b=\frac{\sum (1-T)MF}{\sum (1-T)M+\epsilon},\qquad
z_\Delta=z_p-z_b.
\]

Full 模型融合全局特征、(z_p,z_b,z_\Delta) 与标量条件。网络预测受限对数修正 (delta\in[-0.75,0.75])：

\[
\hat Q=\exp\{\max[0,\log(1+\mu_0)+\delta]\}-1.
\]

同时输出正的非对称半宽 (s_-,s_+>0)，并定义部署风险分数：

\[
r(x)=\frac{s_-(x)+s_+(x)}{2[\hat Q(x)+1]}.
\]

## 4. 损失与 Group-DRO

样本损失为：

\[
\ell=\ell_{Huber}\!\left(\frac{\hat Q-Q}{Q+1};0.30\right)
+0.5\rho_{0.05}\!\left(\frac{Q-(\hat Q-s_-)}{Q+1}\right)
+0.5\rho_{0.95}\!\left(\frac{Q-(\hat Q+s_+)}{Q+1}\right)
+0.01\frac{s_-+s_+}{Q+1}+0.02\delta^2.
\]

背景组损失 (L_g) 为该组样本损失均值。Full、No-Wind 与 No-Contrast 使用 exponentiated-gradient Group-DRO：

\[
w_g^{(t+1)}\propto w_g^{(t)}\exp(0.1L_g^{(t)}),
\]

并把归一化组权重截断到均匀权重的 ([0.25,4]) 倍。No-GDRO 使用均匀组权重。

## 5. 消融（训练前固定）

1. `HPR_FROZEN`：冻结参考，不训练。
2. `WACQ_NO_WIND`：去除 template、顺/横风坐标与前景—背景对比。
3. `WACQ_NO_CONTRAST`：保留 wind/template 通道，但只做全局池化。
4. `WACQ_NO_GDRO`：保留完整架构，使用均匀组权重。
5. `WACQ_FULL`：完整模型，为唯一预注册 primary candidate。

消融只能解释机制，不能在看结果后替换 `WACQ_FULL` 成为主模型。

## 6. 36-background LOBO

严格执行 36 折 Leave-One-Background-Out。每折 35 个背景训练，1 个背景完整外推评价。held-out 背景不用于训练、早停、超参数选择或 conformal 阈值。

每个训练背景固定抽取全部 135 个条件中的 repetition 0 与 25，共 270 行用于神经网络参数拟合；同一训练背景另取 repetition 1、13、26、38，共 540 行，专门用于 LOBO 内 conformal 阈值，且不参与网络拟合。held-out 背景评价全部 6,750 行。固定训练 18 epochs，不做基于 held-out 的 early stopping。

非对称 conformal score 为：

\[
A_i=\max\left\{\frac{\hat Q_i-Q_i}{s_{-,i}},\frac{Q_i-\hat Q_i}{s_{+,i}}\right\}.
\]

每个训练背景独立计算有限样本 90% quantile，外推阈值取 35 个训练背景 quantile 的最大值；这是保守的 Development 级 group guard，不被表述为任意背景迁移下的分布无关保证。

## 7. 预注册 Gate

`WACQ_FULL` 只有同时通过点预测、风险排序和保守区间 Gate 才能称为 `WACQNET_DEVELOPMENT_GATE_PASSED`。主要门槛包括：macro median APE 至少比 HPR 降低 10%，macro (P_{reliable30}) 至少提高 0.05，至少 24/36 背景 median APE 改善；风险 Spearman>0 与 AUPRC>prevalence 各至少 30/36；区间无 PICP<0.80，最多 1 个背景 PICP<0.85，且 macro normalized width 比 frozen A4 至少下降 10%。

这些门槛是训练前冻结的内部工程判据，不是理论普适标准。失败时必须报告 `WACQNET_NOT_ESTABLISHED`，不得用当前或历史 Test 调整模型。

## 8. 后续数据纪律

本阶段成功只意味着 Development Gate 通过。之后仍必须另行预注册并获得全新的 Calibration/Test，才能拟合正式 group-aware conformal 阈值和进行一次性确认评价。现有已经消费的 Test 永远不能用于 WACQNet 选择。
