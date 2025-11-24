# Sharded Optimizer 实现总结

本文档介绍优化器状态分片（Optimizer State Sharding）的实现原理、优点、缺点以及使用方法。

## 概述

Sharded Optimizer 是一种内存优化技术，通过将优化器状态分片到多个设备上，显著减少每个设备的内存占用。这是 ZeRO (Zero Redundancy Optimizer) 的第一阶段 (ZeRO-1) 的简化实现。

## 问题背景

### 标准数据并行训练的内存冗余

在标准的数据并行训练中：
- 每个设备存储完整的模型参数副本
- 每个设备存储完整的梯度副本
- **每个设备存储完整的优化器状态副本**

对于像 AdamW 这样的优化器，每个参数需要 2 个额外的状态（momentum 和 variance），这意味着优化器状态的内存占用是模型权重的 **2 倍**。

**示例**：
- 模型参数：100GB (FP32)
- 梯度：100GB (FP32)
- 优化器状态 (AdamW)：200GB (FP32)
- **总内存/设备**：400GB

如果有 4 个设备，总内存占用为 1600GB，但实际上优化器状态是冗余的（每个设备存储相同的副本）。

### ZeRO 优化

ZeRO (Zero Redundancy Optimizer) 通过消除冗余来减少内存占用：
- **ZeRO-1**：分片优化器状态
- **ZeRO-2**：分片优化器状态 + 梯度
- **ZeRO-3**：分片优化器状态 + 梯度 + 模型参数

本实现对应 **ZeRO-1**。

---

## 实现原理

### 核心思想

1. **参数分片**：
   - 将所有模型参数按 round-robin 方式分配到各个 rank
   - 每个 rank 只负责约 `1/world_size` 的参数

2. **优化器创建**：
   - 每个 rank 只为其负责的参数创建优化器状态
   - 内存占用减少约 `1/world_size`

3. **更新和同步**：
   - 在 `optimizer.step()` 中，每个 rank 只更新其负责的参数
   - 更新后，通过 `broadcast` 操作将更新的参数同步到所有其他 rank

### 关键步骤

1. **初始化** (`__init__`)：
   ```python
   # 收集所有参数
   self._all_params = list(params)
   
   # 分配参数到各个 rank (round-robin)
   for i, param in enumerate(self._all_params):
       assigned_rank = i % self.world_size
       self._param_to_rank[param] = assigned_rank
   
   # 创建底层优化器，只包含本 rank 的参数
   owned_params = [p for p in self._all_params if self._param_to_rank[p] == self.rank]
   self._optimizer = optimizer_cls(owned_params, **kwargs)
   ```

2. **添加参数组** (`add_param_group`)：
   ```python
   # 将新参数分配到各个 rank
   for i, param in enumerate(params):
       assigned_rank = (param_idx + i) % self.world_size
       self._param_to_rank[param] = assigned_rank
   ```

3. **优化器步骤** (`step`)：
   ```python
   # 1. 更新本 rank 的参数
   self._optimizer.step(closure, **kwargs)
   
   # 2. 同步所有参数
   self._synchronize_parameters()
   ```

4. **参数同步** (`_synchronize_parameters`)：
   ```python
   for param in self._all_params:
       owner_rank = self._param_to_rank[param]
       # 从 owner 广播到所有 rank
       dist.broadcast(param.data, src=owner_rank)
   ```

---

## 优点

1. **显著减少内存占用**：
   - 优化器状态内存减少约 `1/world_size`
   - 对于 AdamW，在 4 个设备上可节省约 75% 的优化器状态内存

2. **简单直接**：
   - 实现相对简单，易于理解和调试
   - 不需要修改模型或训练流程

3. **通用性强**：
   - 可以包装任何 PyTorch 优化器（SGD, AdamW, etc.）
   - 与各种 DDP 实现兼容

4. **可扩展性好**：
   - 随着设备数量增加，内存节省线性增加
   - 例如：8 个设备可节省约 87.5% 的优化器状态内存

---

## 缺点

1. **通信开销**：
   - 在每个 `optimizer.step()` 之后需要广播所有参数
   - 通信数据量：`P * 4 bytes`（P 是参数数量，4 是 FP32 字节数）
   - 通信次数：`world_size * num_params` 次 `broadcast` 调用

2. **性能影响**：
   - 增加了训练步骤的总时间
   - 对于小模型或通信带宽有限的场景，可能成为瓶颈

3. **梯度仍然冗余**：
   - 梯度仍然在每个设备上完整存储
   - 如果需要进一步减少内存，需要实现 ZeRO-2 或 ZeRO-3

4. **同步开销**：
   - 参数同步必须在 `optimizer.step()` 之后立即完成
   - 无法与后续计算重叠

---

## 性能分析

### 内存节省

对于一个有 `P` 个参数的模型，使用 AdamW 优化器：

| 项目 | 标准 DDP | Sharded Optimizer | 节省 |
|------|----------|-------------------|------|
| 模型参数 (FP32) | `P * 4` | `P * 4` | 0% |
| 梯度 (FP32) | `P * 4` | `P * 4` | 0% |
| 优化器状态 (FP32) | `P * 8` | `P * 8 / world_size` | ~87.5% (world_size=8) |
| **总计** | `P * 16` | `P * (8 + 8/world_size)` | ~50% (world_size=8) |

### 通信开销

- **标准 DDP (每次迭代)**:
  - 梯度 all-reduce：`P * 4 bytes`（假设 FP32 梯度）
- **Sharded Optimizer (每次迭代)**:
  - 梯度 all-reduce：`P * 4 bytes`
  - 参数 broadcast：`P * 4 bytes`
  - **总通信量翻倍**

### 性能权衡

- **内存受限场景**：Sharded Optimizer 的价值极大，可能是唯一能训练超大模型的方法
- **通信受限场景**：额外的通信开销可能成为瓶颈
- **计算受限场景**：通信开销相对较小，可以接受

---

## 使用建议

### 适用场景

1. **超大模型训练**：
   - 模型参数 > 100B
   - 优化器状态内存是主要瓶颈
   - 例如：GPT-3 (175B)、GPT-4 等

2. **GPU 内存有限**：
   - 无法在单个 GPU 上存储完整的模型 + 梯度 + 优化器状态
   - 需要通过分片来减少内存占用

3. **多设备训练**：
   - 至少 4 个设备（world_size >= 4）
   - 设备越多，内存节省越明显

### 不适用场景

1. **小模型训练**：
   - 模型参数 < 1B
   - 优化器状态内存占用不是瓶颈
   - 额外的通信开销不值得

2. **通信受限环境**：
   - 网络带宽有限
   - 额外的参数广播会显著降低性能

3. **单设备训练**：
   - world_size = 1
   - 没有分片的必要

---

## 与其他方案对比

### vs. 标准 DDP

| 特性 | 标准 DDP | Sharded Optimizer |
|------|----------|-------------------|
| 优化器状态内存 | `P * state_size` | `P * state_size / world_size` |
| 梯度内存 | `P * grad_size` | `P * grad_size` |
| 参数内存 | `P * param_size` | `P * param_size` |
| 通信量/迭代 | `P * grad_size` | `P * (grad_size + param_size)` |
| 实现复杂度 | 低 | 中 |

### vs. ZeRO-2

- **ZeRO-2** 进一步分片梯度，节省更多内存
- **Sharded Optimizer (ZeRO-1)** 只分片优化器状态，实现更简单

### vs. ZeRO-3

- **ZeRO-3** 分片所有内容（参数、梯度、优化器状态）
- **Sharded Optimizer (ZeRO-1)** 只分片优化器状态，通信开销更小

---

## 优化建议

### 1. 通信优化

- **批量广播**：
  - 将多个参数的广播操作合并为单次通信
  - 使用 `all-gather` 替代多次 `broadcast`

- **异步通信**：
  - 使用 `async_op=True` 进行异步广播
  - 与后续计算重叠

### 2. 内存优化

- **结合 Gradient Accumulation**：
  - 减少每个 micro-batch 的内存占用
  - 在多个 micro-batch 后再执行 `optimizer.step()`

- **结合 ZeRO-2**：
  - 进一步分片梯度
  - 进一步减少内存占用

### 3. 性能优化

- **选择合适的分片策略**：
  - Round-robin：简单，负载均衡
  - 按参数大小分片：可能更优，但更复杂

- **监控通信时间**：
  - 如果通信时间 > 计算时间，考虑减少设备数量或使用标准 DDP

---

## 使用示例

### 基本使用

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from cs336_systems.parallel.sharded_optimizer import ShardedOptimizer

# 初始化分布式环境
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# 创建模型
model = MyLargeModel()

# 使用 DDP 进行梯度同步
from cs336_systems.parallel.naive_ddp import NaiveDDP
ddp_model = NaiveDDP(model)

# 创建分片优化器
optimizer = ShardedOptimizer(
    ddp_model.parameters(),
    optimizer_cls=torch.optim.AdamW,
    lr=1e-4,
    weight_decay=0.01
)

# 训练循环
for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # DDP 同步梯度
        ddp_model.sync_gradients()
        
        # 分片优化器更新参数并同步
        optimizer.step()
```

### 与不同 DDP 实现配合

```python
# 与 Flattened DDP 配合
from cs336_systems.parallel.flattened_ddp import FlattenedDDP
ddp_model = FlattenedDDP(model)
optimizer = ShardedOptimizer(ddp_model.parameters(), torch.optim.AdamW, lr=1e-4)

# 与 Overlapped DDP 配合
from cs336_systems.parallel.overlapped_ddp import OverlappedDDP
ddp_model = OverlappedDDP(model)
optimizer = ShardedOptimizer(ddp_model.parameters(), torch.optim.AdamW, lr=1e-4)

# 与 Bucketed DDP 配合
from cs336_systems.parallel.bucketed_ddp import BucketedDDP
ddp_model = BucketedDDP(model, bucket_size_mb=25.0)
optimizer = ShardedOptimizer(ddp_model.parameters(), torch.optim.AdamW, lr=1e-4)
```

---

## 实现细节

### 1. 参数分片策略

使用 **round-robin** 策略将参数分配到各个 rank：

```python
for i, param in enumerate(params):
    assigned_rank = i % world_size
    self._param_to_rank[param] = assigned_rank
```

**优点**：
- 简单，易于实现
- 负载相对均衡

**替代方案**：
- 按参数大小分片：可以更精确地平衡内存和计算
- 按层分片：更符合模型结构，可能有更好的局部性

### 2. 优化器初始化

```python
# 只为本 rank 拥有的参数创建优化器
owned_params = [p for p in all_params if self._param_to_rank[p] == self.rank]
self._optimizer = optimizer_cls(owned_params, **kwargs)
```

**关键点**：
- 底层优化器只看到本 rank 的参数
- 优化器状态只为这些参数创建和存储

### 3. 参数同步

```python
def _synchronize_parameters(self):
    for param in self._all_params:
        owner_rank = self._param_to_rank[param]
        # 从 owner 广播到所有 rank
        dist.broadcast(param.data, src=owner_rank)
```

**通信模式**：
- 每个参数从其 owner rank 广播到所有其他 rank
- 通信次数：`num_params` 次 `broadcast`
- 总通信量：`P * 4 bytes`（P 是参数数量）

**优化方向**：
- 使用 `all-gather` 替代多次 `broadcast`
- 批量同步多个参数

### 4. 梯度处理

```python
def zero_grad(self, set_to_none: bool = True):
    # 清零所有参数的梯度（不仅仅是本 rank 的）
    for param in self._all_params:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()
```

**关键点**：
- 梯度在 `backward()` 时为所有参数计算
- 需要清零所有参数的梯度，而不仅仅是本 rank 拥有的

---

## 性能对比

### 内存占用

假设模型有 220B 参数，使用 AdamW 优化器，world_size=8：

| 项目 | 标准 DDP | Sharded Optimizer | 节省 |
|------|----------|-------------------|------|
| 模型参数 (FP32) | 880 GB | 880 GB | 0% |
| 梯度 (FP32) | 880 GB | 880 GB | 0% |
| 优化器状态 (FP32) | 1760 GB | 220 GB | 87.5% |
| **总计/设备** | 3520 GB | 1980 GB | 43.8% |

### 通信开销

| 操作 | 标准 DDP | Sharded Optimizer | 增加 |
|------|----------|-------------------|------|
| 梯度同步 | `P * 4` bytes | `P * 4` bytes | 0% |
| 参数同步 | 0 | `P * 4` bytes | +100% |
| **总通信量/迭代** | `P * 4` bytes | `P * 8` bytes | +100% |

### 性能影响

- **大模型 (> 100B)**：
  - 内存节省显著，值得额外的通信开销
  - 通常能实现 1.5-2x 的模型大小扩展

- **中型模型 (10B - 100B)**：
  - 内存节省适中，需要权衡通信开销
  - 如果内存充足，标准 DDP 可能更快

- **小模型 (< 10B)**：
  - 内存节省不明显，通信开销相对更大
  - 不推荐使用

---

## 与 DDP 的配合

Sharded Optimizer 通常与 DDP 配合使用：

1. **梯度同步**：由 DDP 处理（all-reduce）
2. **参数更新**：由 Sharded Optimizer 处理（分片更新 + broadcast 同步）

**训练流程**：

```
1. Forward pass
2. Backward pass (计算梯度)
3. DDP 同步梯度 (all-reduce)
4. Sharded Optimizer 更新参数 (每个 rank 更新其分片)
5. Sharded Optimizer 同步参数 (broadcast)
```

---

## 扩展方向

### 1. ZeRO-2: 分片梯度

- 在 `backward()` 后，使用 `reduce-scatter` 将梯度分片
- 每个 rank 只保留其负责参数的梯度
- 进一步减少梯度内存占用

### 2. ZeRO-3: 分片参数

- 将模型参数也分片
- 在 `forward()` 前使用 `all-gather` 收集参数
- 在 `forward()` 后释放非本 rank 的参数
- 最大化内存节省，但通信开销最大

### 3. 通信优化

- **批量广播**：将多个参数合并为单个张量后再广播
- **异步通信**：使用 `async_op=True` 进行异步广播
- **通信/计算重叠**：在同步参数的同时准备下一个 batch

---

## 总结

Sharded Optimizer 是一种有效的内存优化技术，特别适合训练超大规模模型。它通过将优化器状态分片到多个设备上，显著减少了每个设备的内存占用，代价是增加了约 100% 的通信量。

**关键要点**：
- 内存节省：优化器状态内存减少约 `1/world_size`
- 通信开销：每次迭代的通信量翻倍
- 适用场景：大模型、内存受限、多设备训练

**推荐配置**：
- 模型大小 > 100B 参数
- 设备数量 >= 4
- 通信带宽充足
- 与 Bucketed DDP 或 Overlapped DDP 配合使用

---

