# Nsight Systems (nsys) 使用指南

## 代码修改说明

代码已经添加了 NVTX 标注，用于标记不同的执行阶段：

1. **warmup** - 标记所有 warm-up 步骤（可以在 nsys 中过滤掉）
2. **step_N** - 标记每个测量步骤
3. **forward_pass** - 标记前向传播
4. **backward_pass** - 标记后向传播

## 基本使用方法

### 1. 运行 nsys 分析（前向+后向）

```bash
uv run nsys profile \
    -o result \
    --trace=cuda,nvtx \
    python cs336_systems/benchmark/benchmark.py \
        --context_length 512 \
        --d_model 768 \
        --num_layers 12 \
        --num_heads 12 \
        --d_ff 3072 \
        --warmup_steps 5 \
        --num_steps 10
```

### 2. 只分析前向传播

```bash
uv run nsys profile \
    -o result_forward \
    --trace=cuda,nvtx \
    python cs336_systems/benchmark/benchmark.py \
        --context_length 512 \
        --d_model 768 \
        --num_layers 12 \
        --num_heads 12 \
        --d_ff 3072 \
        --warmup_steps 5 \
        --num_steps 10 \
        --forward_only
```

### 3. 带 CUDA 调用栈的分析

```bash
uv run nsys profile \
    -o result \
    --trace=cuda,nvtx \
    --cudabacktrace=true \
    python cs336_systems/benchmark/benchmark.py \
        --context_length 512 \
        --d_model 768 \
        --num_layers 12 \
        --num_heads 12 \
        --d_ff 3072 \
        --warmup_steps 5 \
        --num_steps 10
```

## 命令行选项说明

### nsys profile 选项

- `-o result`: 输出文件名（会生成 `result.qdrep`）
- `--trace=cuda,nvtx`: 追踪 CUDA API 和 NVTX 事件（代码中已手动添加 NVTX 标注）
- `--cudabacktrace=true`: 为每个 CUDA API 调用添加调用栈（可能有性能开销）

### benchmark.py 选项

- `--forward_only`: 只运行前向传播
- 其他选项与之前相同

## 查看报告

### 方法 1: 使用 Nsight Systems GUI

```bash
# 如果有图形界面
nsys-ui result.qdrep

# 或者下载到本地后用 Nsight Systems 打开
```

### 方法 2: 使用命令行查看统计信息

```bash
# 查看基本统计
nsys stats result.qdrep

# 查看 GPU 跟踪
nsys stats --report gputrace result.qdrep

# 导出为 CSV
nsys stats --report gputrace --format csv result.qdrep > result.csv
```

## 在 Nsight Systems GUI 中分析

### 过滤 warm-up 步骤

1. 打开 `result.qdrep` 文件
2. 在 NVTX 行中找到 "warmup" 范围
3. 使用时间轴过滤器，排除 "warmup" 范围
4. 只查看 "step_0", "step_1" 等测量步骤

### 查看前向/后向步骤

1. 在 NVTX 行中可以看到：
   - `forward_pass` - 前向传播
   - `backward_pass` - 后向传播

2. 选择每个范围，查看对应的 CUDA 内核执行时间

3. 在 CUDA API 行选择调用，会在 CUDA HW 行高亮对应的内核

### 获取时间信息

1. 选择 NVTX 范围（如 "forward_pass"）
2. 查看时间轴上的持续时间
3. 或者在统计面板中查看累计时间

## 批量测试脚本示例

```bash
#!/bin/bash
# 测试所有模型大小和上下文长度

MODELS=(
    "small:768:3072:12:12"
    "medium:1024:4096:24:16"
    "large:1280:5120:36:20"
    "xl:1600:6400:48:25"
    "2.7B:2560:10240:32:32"
)

CONTEXT_LENGTHS=(128 256 512 1024)

for model_info in "${MODELS[@]}"; do
    IFS=':' read -r name d_model d_ff num_layers num_heads <<< "$model_info"
    
    for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
        echo "Testing $name with context_length=$ctx_len"
        
        uv run nsys profile \
            -o "result_${name}_ctx${ctx_len}" \
            --trace=cuda,nvtx \
            python cs336_systems/benchmark/benchmark.py \
                --context_length $ctx_len \
                --d_model $d_model \
                --num_layers $num_layers \
                --num_heads $num_heads \
                --d_ff $d_ff \
                --warmup_steps 5 \
                --num_steps 10 || echo "Failed: $name ctx=$ctx_len (possibly OOM)"
    done
done
```

## 注意事项

1. **内存限制**：大模型（如 2.7B）在长上下文（1024）下可能内存不足，这是正常的
2. **文件大小**：nsys 报告文件可能很大（几 GB），确保有足够磁盘空间
3. **性能开销**：使用 `--cudabacktrace=true` 会增加性能开销，可能影响测量结果
4. **过滤 warm-up**：在 GUI 中记得过滤掉 warm-up 步骤，只分析实际测量步骤
5. **NVTX 标注**：代码中已手动添加 NVTX 标注，无需 `--pytorch` 选项（某些版本的 nsys 不支持此选项）

