# MPFormer 代码学习导图

这份笔记的目标是把论文里的概念落到代码文件上，帮助你按模块学习这个项目。

论文题目：Multi-scale Physics-informed Transformer With Spatio-temporal Feature Adapter For Extreme Precipitation Nowcasting

## 1. 先看整体入口

推荐从这几个文件开始：

1. `code/run.py`
   - 训练入口。
   - 定义数据长度、图像大小、通道数、batch size、学习率、checkpoint 路径等超参数。
   - 默认输入 9 帧，预测总长度 29 帧，因此预测未来 20 帧。

2. `code/test.py`
   - 测试/验证入口。
   - 自动找 checkpoint 目录下最新的 `.ckpt`。
   - 调用 `evaluator.test_pytorch_loader2` 保存预测图像。

3. `code/mpformer/train.py`
   - 创建 DataLoader。
   - 构建 Lightning 模型 `Model`。
   - 配置 checkpoint、logger、Trainer。

4. `code/mpformer/models/model_factory.py`
   - LightningModule 外壳。
   - 负责训练步骤、验证步骤、损失函数、指标统计、优化器。

5. `code/mpformer/models/mpformer.py`
   - 真正的 MPFormer 主体。
   - 把论文里的物理演化分支、生成分支、噪声投影、adapter 串起来。

## 2. 数据格式

数据加载在：

- `code/mpformer/data_provider/datasets_factory.py`
- `code/mpformer/data_provider/loader.py`

目录组织假设如下：

```text
data_path/
  case_name_1/
    case_name_1-00.png
    case_name_1-01.png
    ...
    case_name_1-28.png
  case_name_2/
    case_name_2-00.png
    ...
```

每个样本固定读取 29 张 PNG。

`loader.py` 输出：

```text
radar_frames: [T, H, W, 2]
```

两个通道含义：

- `[..., 0]`：雷达降水强度，经过 `/10 - 3`、小于 0 置 0、clip 到 `[0, 128]`
- `[..., 1]`：mask，原始值小于 0 的地方为 0，其余为 1

训练时 batch 后变成：

```text
[B, T, H, W, 2]
```

其中默认：

```text
B = batch_size
T = 29
H = W = 512
C = 2
```

## 3. 论文概念和代码对应

### 3.1 Multi-scale Transformer / SMT

代码位置：

- `code/mpformer/layers/smt/smtseg.py`
- `code/mpformer/layers/smt/model/smt.py`
- `code/mpformer/layers/smt/model/decoder.py`

`SMTWrapper` 包了一套 SMT backbone 加 UPerHead decoder：

```python
self.backbone = SMT(...)
self.decode_head = UPerHead(...)
```

在主模型里，SMT 被用于物理演化网络：

```python
self.evo_net = Evolution_Network(...)
```

`Evolution_Network` 默认参数 `net="smt"`，所以实际走的是 SMT 分支。

### 3.2 Physics-informed Evolution Network

代码位置：

- `code/mpformer/layers/evolution/evolution_network.py`
- `code/mpformer/layers/utils.py`
- `code/mpformer/models/mpformer.py`

这部分是论文里“物理启发/物理约束”的核心代码落点。

`Evolution_Network.forward()` 输出两个量：

```python
intensity, motion = self.evo_net(input_frames)
```

含义可以这样理解：

- `intensity`：未来每个时刻的强度增量，形状最终整理为 `[B, pred_length, 1, H, W]`
- `motion`：未来每个时刻的二维运动场，形状最终整理为 `[B, pred_length, 2, H, W]`

然后 `mpformer.py` 里用 `warp()` 做逐步外推：

```python
last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")
last_frames = last_frames + intensity_[:, i]
```

这就是一个典型的“平流/运动场外推 + 强度残差修正”过程：

```text
下一帧 = 按运动场扭曲上一帧 + 强度变化项
```

这一步得到 `evo_result`，它是后续生成网络的条件输入。

### 3.3 Spatio-temporal Feature Adapter

代码位置：

- `code/mpformer/layers/adapters/bottleneck_adapter.py`
- `code/mpformer/layers/evolution/evolution_network.py`
- `code/mpformer/layers/generation/generative_network.py`

adapter 是一个很轻量的瓶颈残差模块：

```python
down_proj: C -> bottleneck_dim
ReLU
up_proj: bottleneck_dim -> C
residual add
```

对应代码：

```python
return x + residual
```

在这个项目里 adapter 被加在两处：

1. 演化网络的 SMT 前：

```python
x1 = self.inc(x)
x2 = self.adapter(x1)
x3 = self.smt(x2)
```

2. 生成编码器的多个下采样层后：

```python
x = self.down1(x)
x = self.adapters[0](x)
x = self.down2(x)
x = self.adapters[1](x)
x = self.down3(x)
x = self.adapters[2](x)
```

`configs.adapter=True` 时，`mpformer.py` 会冻结非 adapter 参数：

```python
if self.configs.adapter:
    self._freeze_backbone()
```

所以这里的 adapter 更像论文里的轻量微调/特征适配模块。

注意：如果没有加载预训练权重，同时又打开 `adapter=True`，大部分随机初始化的主干会被冻结，这在训练上通常是不合理的。学习代码时要特别留意 `run.py` 里的：

```python
--adapter default=True
--pretrained_model default='mpformer.ckpt'
```

## 4. 主模型 forward 数据流

代码位置：

- `code/mpformer/models/mpformer.py`

主流程可以按下面理解：

```text
输入 all_frames
  shape: [B, 9 或 29, H, W, 2]

只取第 0 通道降水强度
  -> [B, T, H, W, 1]

整理为 input_frames
  -> [B, 9, H, W]

Evolution_Network
  -> intensity: [B, 20, H, W]
  -> motion:    [B, 40, H, W]

循环 20 次：
  上一帧按 motion warp
  再加 intensity
  得到 evo_result: [B, 20, H, W]

拼接历史输入和物理演化结果：
  cat([input_frames, evo_result], dim=1)
  -> [B, 29, H, W]

Generative_Encoder
  -> evo_feature

随机噪声 noise
  -> Noise_Projector
  -> noise_feature

拼接 evo_feature 和 noise_feature
  -> Generative_Decoder
  -> gen_result: [B, 20, H, W]

最后 unsqueeze
  -> [B, 20, H, W, 1]
```

可以把主模型理解成两阶段：

1. 先产生一个符合运动场外推逻辑的粗预测 `evo_result`
2. 再用生成网络结合多尺度特征、噪声和 SPADE 条件归一化生成最终预测

## 5. 生成网络

代码位置：

- `code/mpformer/layers/generation/generative_network.py`
- `code/mpformer/layers/generation/module.py`
- `code/mpformer/layers/generation/noise_projector.py`

### 5.1 Generative_Encoder

输入：

```text
[B, 29, H, W]
```

也就是：

```text
9 帧真实历史 + 20 帧物理演化粗预测
```

经过三次下采样，提取多尺度空间特征。

### 5.2 Noise_Projector

随机噪声：

```python
noise = torch.randn(batch, ngf, height // 32, width // 32)
```

然后通过多层 `ProjBlock` 投影，再 reshape 到和生成特征可拼接的尺度。

### 5.3 Generative_Decoder + SPADE

`Generative_Decoder` 里大量使用 `GenBlock`。

`GenBlock` 的关键在 `SPADE`：

```python
self.norm_0 = SPADE(fin, ic)
```

其中 `evo_result` 作为条件输入，参与归一化参数 `gamma/beta` 的生成：

```python
out = normalized * (1 + gamma) + beta
```

直观理解：

```text
生成网络不是自由生成，而是被物理演化结果 evo_result 调制。
```

这也是代码里“physics-informed”和“generative refinement”结合的地方。

## 6. 损失函数

代码位置：

- `code/mpformer/models/model_factory.py`
- `code/mpformer/utils/loss_assemble.py`

训练实际使用：

```python
self.criterion = SASTLoss(losstype='single')
```

`SASTLoss` 由四部分组成：

```text
BMSELoss + BMAELoss + WSloss_linear_add_adhoc + WTloss
```

对应作用：

- `BMSELoss`：按降水阈值加权的 MSE
- `BMAELoss`：按降水阈值加权的 MAE
- `WSloss_linear_add_adhoc`：小波多尺度 SSIM 损失
- `WTloss`：小波多尺度 L1 损失

这和论文里的 multi-scale 思想相对应：不仅看像素误差，也看不同频率/尺度上的结构误差。

学习时注意两点：

1. `CustomLoss`、`get_prototypes()`、`get_pairs()` 看起来像实验残留，目前训练实际没有使用 `CustomLoss`。
2. 损失函数里有不少 `.cuda()`，所以代码默认强依赖 CUDA。

## 7. 指标和验证

代码位置：

- `code/mpformer/utils/metrics.py`
- `code/mpformer/evaluator.py`
- `code/mpformer/evaluator_time.py`

训练/验证时记录：

- HSS
- Neighbourhood CSI
- PSD
- RMSE
- CRPS
- FSS
- MAE

`evaluator.py` 主要负责把预测结果保存成图片。

`evaluator_time.py` 更偏向逐 lead time 统计指标。

## 8. 推荐学习顺序

第一遍只看数据流：

1. `run.py`
2. `train.py`
3. `model_factory.py`
4. `mpformer.py`
5. `loader.py`

目标：弄清楚输入输出 shape。

第二遍看论文核心模块：

1. `evolution_network.py`
2. `smtseg.py`
3. `smt.py`
4. `bottleneck_adapter.py`
5. `generative_network.py`
6. `module.py`

目标：把论文里的 transformer、adapter、physics-informed evolution 和生成网络对上。

第三遍看训练目标：

1. `loss_assemble.py`
2. `metrics.py`
3. `evaluator_time.py`

目标：理解模型为什么会偏向强降水、结构相似性和多尺度细节。

## 9. 读代码时最重要的三个问题

### 问题 1：模型为什么要先做 evolution？

因为极端降水临近预报不是纯图像生成问题，降水回波有明显的运动连续性。

代码用 `motion` 表示运动场，用 `intensity` 表示强度变化：

```text
future = warp(previous, motion) + intensity
```

这相当于把物理先验嵌入网络预测过程。

### 问题 2：生成网络为什么还要存在？

`evo_result` 是粗预测，能提供运动趋势，但可能不够清晰，也可能缺少局地强对流细节。

生成网络负责在多尺度特征和随机噪声帮助下细化结果，同时通过 SPADE 被 `evo_result` 约束。

### 问题 3：adapter 在这里学什么？

adapter 学的是轻量的特征修正。

如果主干网络已有预训练权重，冻结主干、训练 adapter 可以减少训练成本，也可以让模型适应新的雷达数据分布。

## 10. 学习时可以画的一张图

```text
历史 9 帧雷达
     |
     v
Evolution_Network(SMT + Adapter)
     |                      |
     |                      +--> motion field
     +--------------------------> intensity change
     |
     v
warp + intensity
     |
     v
evo_result 粗预测 20 帧
     |
     +----------------------+
                            |
历史 9 帧 + evo_result 20 帧 |
     |                      |
     v                      |
Generative_Encoder          |
     |                      |
     +-- concat noise_feature
     |
     v
Generative_Decoder(SPADE conditioned by evo_result)
     |
     v
最终预测 20 帧
```

## 11. 这个仓库里值得警惕的实现细节

这些不是你理解论文的主线，但后续跑代码时很可能会遇到：

1. `requirements.txt` 不是标准 pip requirements，更像 conda 导出文件，而且里面包含 Linux 包名。
2. 代码大量写死 `.cuda()`，在 CPU 或非默认 GPU 环境容易报错。
3. `run.py` 默认 `adapter=True`，如果没有正确加载预训练权重，可能会冻结随机初始化主干。
4. `datasets_factory.test_data_provider()` 里有一处疑似 bug：`datasets_map[configs.dataset_path_test]` 应该更可能是 `datasets_map[configs.dataset_name]`。
5. 部分中文注释编码损坏，不影响运行，但影响阅读。
6. `CustomLoss` 和 prototype/contrastive 相关函数目前像是未启用实验代码。

