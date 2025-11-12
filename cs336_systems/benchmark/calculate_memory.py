#!/usr/bin/env python3
"""
计算2.7B模型在训练时的内存占用（float32）
"""

def calculate_model_memory(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    bytes_per_param: int = 4,  # float32 = 4 bytes
):
    """计算模型参数的内存占用"""
    
    # Embedding层
    embedding_params = vocab_size * d_model
    
    # 每个Transformer Block的参数
    d_head = d_model // num_heads
    
    # Attention层参数
    # QKV投影: d_model * (3 * d_model) = 3 * d_model^2
    # 输出投影: d_model * d_model = d_model^2
    attn_params_per_layer = 3 * d_model * d_model + d_model * d_model  # 4 * d_model^2
    
    # FFN层参数 (SwiGLU: w1, w2, w3)
    # w1: d_model * d_ff
    # w2: d_ff * d_model
    # w3: d_model * d_ff
    ffn_params_per_layer = d_model * d_ff + d_ff * d_model + d_model * d_ff  # 3 * d_model * d_ff
    
    # LayerNorm参数 (每个block有2个，每个d_model个参数)
    ln_params_per_layer = 2 * d_model
    
    # 每个Transformer Block的总参数
    block_params = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    
    # 所有Transformer Blocks的参数
    all_blocks_params = num_layers * block_params
    
    # 最终LayerNorm
    final_ln_params = d_model
    
    # LM Head (输出层)
    lm_head_params = d_model * vocab_size
    
    # 总参数数
    total_params = embedding_params + all_blocks_params + final_ln_params + lm_head_params
    
    # 总内存（字节）
    total_memory_bytes = total_params * bytes_per_param
    
    return {
        'embedding_params': embedding_params,
        'attn_params_per_layer': attn_params_per_layer,
        'ffn_params_per_layer': ffn_params_per_layer,
        'ln_params_per_layer': ln_params_per_layer,
        'block_params': block_params,
        'all_blocks_params': all_blocks_params,
        'final_ln_params': final_ln_params,
        'lm_head_params': lm_head_params,
        'total_params': total_params,
        'total_memory_bytes': total_memory_bytes,
        'total_memory_mb': total_memory_bytes / (1024 ** 2),
        'total_memory_gb': total_memory_bytes / (1024 ** 3),
    }


def calculate_activation_memory(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    vocab_size: int,
    bytes_per_element: int = 4,  # float32 = 4 bytes
    use_gradient_checkpointing: bool = False,
):
    """计算激活值的内存占用（前向传播）
    
    如果不使用梯度检查点，需要保存所有层的激活值用于反向传播。
    如果使用梯度检查点，只需要保存当前层的激活值。
    """
    
    # 输入embedding激活: batch_size * seq_len * d_model
    input_embedding = batch_size * seq_len * d_model
    
    # 每层需要保存的激活值（用于反向传播）：
    # 1. 层输入 (用于残差连接): batch_size * seq_len * d_model
    layer_input = batch_size * seq_len * d_model
    
    # 2. 注意力输出前的激活值 (用于残差连接): batch_size * seq_len * d_model
    attn_before_residual = batch_size * seq_len * d_model
    
    # 3. 注意力矩阵 (用于softmax的梯度): batch_size * num_heads * seq_len * seq_len
    attention_matrix = batch_size * num_heads * seq_len * seq_len
    
    # 4. FFN输入 (用于残差连接): batch_size * seq_len * d_model
    ffn_input = batch_size * seq_len * d_model
    
    # 5. FFN中间激活值 (SwiGLU的输出，用于反向传播): batch_size * seq_len * d_ff
    ffn_intermediate = batch_size * seq_len * d_ff
    
    # 每层的激活值总数
    per_layer_activation = (
        layer_input + 
        attn_before_residual + 
        attention_matrix + 
        ffn_input + 
        ffn_intermediate
    )
    
    # 输出logits: batch_size * seq_len * vocab_size
    output_logits = batch_size * seq_len * vocab_size
    
    if use_gradient_checkpointing:
        # 使用梯度检查点：只需要保存当前层的激活值
        peak_activation = input_embedding + per_layer_activation + output_logits
        total_activation = peak_activation
    else:
        # 不使用梯度检查点：需要保存所有层的激活值
        all_layers_activation = num_layers * per_layer_activation
        total_activation = input_embedding + all_layers_activation + output_logits
        # 峰值激活值（前向传播过程中，同时存在输入和当前层的激活值）
        peak_activation = input_embedding + per_layer_activation + output_logits
    
    return {
        'input_embedding': input_embedding,
        'per_layer_activation': per_layer_activation,
        'attention_matrix': attention_matrix,
        'ffn_intermediate': ffn_intermediate,
        'peak_activation': peak_activation,
        'total_activation': total_activation,
        'peak_activation_mb': peak_activation * bytes_per_element / (1024 ** 2),
        'total_activation_mb': total_activation * bytes_per_element / (1024 ** 2),
        'peak_activation_gb': peak_activation * bytes_per_element / (1024 ** 3),
        'total_activation_gb': total_activation * bytes_per_element / (1024 ** 3),
    }


def calculate_training_memory(
    model_config: dict,
    batch_size: int = 4,
    seq_len: int = 256,
    use_optimizer: bool = True,
    bytes_per_param: int = 4,
):
    """计算完整训练时的内存占用"""
    
    # 模型参数内存
    model_mem = calculate_model_memory(**model_config, bytes_per_param=bytes_per_param)
    
    # 梯度内存（与参数相同大小）
    gradient_memory_bytes = model_mem['total_params'] * bytes_per_param
    gradient_memory_gb = gradient_memory_bytes / (1024 ** 3)
    
    # 优化器状态内存（AdamW需要momentum和variance，各为参数大小）
    optimizer_memory_bytes = 0
    if use_optimizer:
        # AdamW: momentum + variance = 2 * 参数大小
        optimizer_memory_bytes = 2 * model_mem['total_params'] * bytes_per_param
    optimizer_memory_gb = optimizer_memory_bytes / (1024 ** 3)
    
    # 激活值内存（不使用梯度检查点的情况）
    activation_mem = calculate_activation_memory(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        num_layers=model_config['num_layers'],
        vocab_size=model_config['vocab_size'],
        bytes_per_element=bytes_per_param,
        use_gradient_checkpointing=False,
    )
    
    # 总内存占用
    # 注意：激活值内存在前向和后向传播中会变化
    # 不使用梯度检查点时，需要保存所有层的激活值
    # 使用梯度检查点时，只需要保存当前层的激活值
    total_memory_bytes = (
        model_mem['total_memory_bytes'] +  # 模型参数
        gradient_memory_bytes +  # 梯度
        optimizer_memory_bytes +  # 优化器状态
        activation_mem['total_activation'] * bytes_per_param  # 所有激活值（不使用checkpointing）
    )
    
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    
    return {
        'model_params': model_mem,
        'gradient_memory_gb': gradient_memory_gb,
        'optimizer_memory_gb': optimizer_memory_gb,
        'activation_memory': activation_mem,
        'total_memory_gb': total_memory_gb,
        'breakdown': {
            'model_weights_gb': model_mem['total_memory_gb'],
            'gradients_gb': gradient_memory_gb,
            'optimizer_state_gb': optimizer_memory_gb,
            'peak_activations_gb': activation_mem['peak_activation_gb'],
        }
    }


def main():
    # 2.7B模型配置
    model_config = {
        'vocab_size': 10000,
        'd_model': 2560,
        'num_layers': 32,
        'num_heads': 32,
        'd_ff': 10240,
    }
    
    batch_size = 4
    context_length = 512
    
    print("=" * 80)
    print("2.7B模型内存占用计算 (float32)")
    print("=" * 80)
    print(f"\n模型配置:")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  d_ff: {model_config['d_ff']}")
    print(f"  num_layers: {model_config['num_layers']}")
    print(f"  num_heads: {model_config['num_heads']}")
    print(f"  vocab_size: {model_config['vocab_size']}")
    print(f"\n训练配置:")
    print(f"  batch_size: {batch_size}")
    print(f"  context_length: {context_length}")
    print(f"  精度: float32 (4 bytes/param)")
    
    # 计算训练内存
    training_mem = calculate_training_memory(
        model_config=model_config,
        batch_size=batch_size,
        seq_len=context_length,
        use_optimizer=True,
        bytes_per_param=4,
    )
    
    print("\n" + "=" * 80)
    print("内存占用明细")
    print("=" * 80)
    
    print(f"\n1. 模型参数:")
    print(f"   总参数数: {training_mem['model_params']['total_params']:,}")
    print(f"   内存占用: {training_mem['model_params']['total_memory_gb']:.2f} GB")
    
    print(f"\n2. 梯度内存:")
    print(f"   内存占用: {training_mem['gradient_memory_gb']:.2f} GB")
    
    print(f"\n3. 优化器状态 (AdamW):")
    print(f"   内存占用: {training_mem['optimizer_memory_gb']:.2f} GB")
    print(f"   (momentum + variance = 2 × 参数大小)")
    
    print(f"\n4. 激活值内存 (峰值):")
    print(f"   峰值激活值: {training_mem['activation_memory']['peak_activation']:,} 元素")
    print(f"   内存占用: {training_mem['activation_memory']['peak_activation_gb']:.2f} GB")
    
    print(f"\n5. 总内存占用 (峰值):")
    print(f"   {training_mem['total_memory_gb']:.2f} GB")
    
    print("\n" + "=" * 80)
    print("内存占用分解")
    print("=" * 80)
    breakdown = training_mem['breakdown']
    print(f"  模型权重:     {breakdown['model_weights_gb']:>8.2f} GB")
    print(f"  梯度:         {breakdown['gradients_gb']:>8.2f} GB")
    print(f"  优化器状态:   {breakdown['optimizer_state_gb']:>8.2f} GB")
    print(f"  峰值激活值:   {breakdown['peak_activations_gb']:>8.2f} GB")
    print(f"  {'-' * 40}")
    print(f"  总计:         {training_mem['total_memory_gb']:>8.2f} GB")
    
    print("\n" + "=" * 80)
    print("注意事项")
    print("=" * 80)
    print("1. 激活值内存是峰值估计，实际值取决于是否使用梯度检查点")
    print("2. 如果使用梯度检查点，激活值内存会显著减少")
    print("3. 实际训练中可能还需要额外的临时缓冲区内存")
    print("4. 这里计算的是单GPU训练的内存占用")


if __name__ == "__main__":
    main()

