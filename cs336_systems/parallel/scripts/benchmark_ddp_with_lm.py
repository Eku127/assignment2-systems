#!/usr/bin/env python3
"""
Benchmark script for Large model with different DDP implementations.

This script benchmarks the Large language model using different DDP implementations
to measure training step time and gradient communication overhead.

Model Configuration (Large):
  - d_model: 1280
  - num_layers: 36
  - num_heads: 20
  - d_ff: 5120
  - vocab_size: 10000
  - context_length: 512
  - batch_size: 4 (per process)

Usage:
    # Run with naive DDP (default)
    uv run python -m cs336_systems.parallel.scripts.benchmark_xl_ddp --ddp_type naive
    
    # Run with other DDP implementations (when available)
    uv run python -m cs336_systems.parallel.scripts.benchmark_xl_ddp --ddp_type bucketed
"""

import argparse
import os
import tempfile
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from typing import Optional

# Import language model
import cs336_basics.model as basics_model

# Import DDP implementations
from cs336_systems.parallel.naive_ddp import NaiveDDP
from cs336_systems.parallel.flattened_ddp import FlattenedDDP
from cs336_systems.parallel.overlapped_ddp import OverlappedDDP
from cs336_systems.parallel.bucketed_ddp import BucketedDDP


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize the process group for distributed communication.
    
    Args:
        rank: The rank of the current process (0 to world_size-1)
        world_size: Total number of processes in the group
        backend: Backend to use ("gloo" for CPU, "nccl" for GPU)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"  # Use different port to avoid conflicts
    
    # For NCCL backend, set each process to use a different GPU
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but CUDA is not available")
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"NCCL backend requires at least {world_size} GPUs, "
                f"but only {torch.cuda.device_count()} GPUs are available"
            )
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_distributed():
    """Clean up the process group."""
    dist.barrier()
    dist.destroy_process_group()


def create_ddp_model(model: nn.Module, ddp_type: str = "naive", bucket_size_mb: float = 25.0, **kwargs) -> nn.Module:
    """Create a DDP-wrapped model based on the specified type.
    
    Args:
        model: The base model to wrap
        ddp_type: Type of DDP implementation ("naive", "flattened", "overlapped", "bucketed")
        bucket_size_mb: Bucket size in MB (only used for bucketed DDP, default: 25.0)
        **kwargs: Additional arguments for specific DDP implementations
    
    Returns:
        DDP-wrapped model
    """
    if ddp_type == "naive":
        return NaiveDDP(model)
    elif ddp_type == "flattened":
        return FlattenedDDP(model)
    elif ddp_type == "overlapped":
        return OverlappedDDP(model)
    elif ddp_type == "bucketed":
        return BucketedDDP(model, bucket_size_mb=bucket_size_mb)
    else:
        raise ValueError(f"Unknown DDP type: {ddp_type}")


def sync_gradients_ddp(model: nn.Module, ddp_type: str = "naive"):
    """Synchronize gradients based on DDP type.
    
    Args:
        model: DDP-wrapped model
        ddp_type: Type of DDP implementation
    """
    if ddp_type == "naive":
        if isinstance(model, NaiveDDP):
            model.sync_gradients()
        else:
            raise ValueError("Model is not a NaiveDDP instance")
    elif ddp_type == "flattened":
        if isinstance(model, FlattenedDDP):
            model.sync_gradients()
        else:
            raise ValueError("Model is not a FlattenedDDP instance")
    elif ddp_type == "overlapped":
        if isinstance(model, OverlappedDDP):
            model.finish_gradient_synchronization()
        else:
            raise ValueError("Model is not an OverlappedDDP instance")
    elif ddp_type == "bucketed":
        if isinstance(model, BucketedDDP):
            model.finish_gradient_synchronization()
        else:
            raise ValueError("Model is not a BucketedDDP instance")
    else:
        raise ValueError(f"Unknown DDP type: {ddp_type}")


def train_step_ddp(
    rank: int,
    world_size: int,
    ddp_type: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    batch_size: int,
    num_steps: int,
    warmup_steps: int,
    backend: str,
    bucket_size_mb: float = 25.0,
    result_file: Optional[str] = None,
):
    """Run training steps with DDP and measure timing.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        ddp_type: Type of DDP implementation
        vocab_size: Vocabulary size
        context_length: Context length (sequence length)
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        batch_size: Batch size per process
        num_steps: Number of training steps to measure
        warmup_steps: Number of warmup steps
        backend: Distributed backend
        result_file: Path to save timing results (only rank 0 writes)
    """
    device = setup_distributed(rank, world_size, backend)
    
    # Create model
    model = basics_model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
    )
    model = model.to(device)
    
    # Wrap with DDP
    ddp_model = create_ddp_model(model, ddp_type=ddp_type, bucket_size_mb=bucket_size_mb)
    
    # Create optimizer
    # Note: AdamW requires 2x parameter memory (momentum + variance)
    # For memory-constrained scenarios, consider using SGD instead
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)  # Saves ~14.89 GB
    
    # Generate random data (each rank gets different data)
    torch.manual_seed(42 + rank)  # Different seed per rank
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    
    # Timing statistics
    iteration_times = []
    communication_times = []
    
    # Warm-up steps
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        outputs = ddp_model(input_ids)
        # Use cross-entropy loss (sum over sequence and batch, then mean)
        loss = outputs.view(-1, vocab_size).log_softmax(dim=-1)
        # Create dummy targets (shifted input_ids)
        targets = input_ids.view(-1)
        loss = nn.functional.nll_loss(loss, targets, reduction='mean')
        loss.backward()
        
        # Synchronize gradients
        sync_gradients_ddp(ddp_model, ddp_type=ddp_type)
        
        optimizer.step()
        
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Actual timing measurements
    for step in range(num_steps):
        # Synchronize before timing (for GPU)
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        iteration_start = time.time()
        
        optimizer.zero_grad()
        outputs = ddp_model(input_ids)
        # Use cross-entropy loss
        loss = outputs.view(-1, vocab_size).log_softmax(dim=-1)
        targets = input_ids.view(-1)
        loss = nn.functional.nll_loss(loss, targets, reduction='mean')
        loss.backward()
        
        # Measure communication time
        # For overlapped DDP, communication happens during backward pass
        # We measure the time to finish synchronization (waiting for async ops)
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
        comm_start = time.time()
        
        # Synchronize gradients
        # For overlapped DDP, this waits for all async all-reduces to complete
        sync_gradients_ddp(ddp_model, ddp_type=ddp_type)
        
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
        comm_end = time.time()
        
        optimizer.step()
        
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
        iteration_end = time.time()
        
        # Record timings
        iteration_time = iteration_end - iteration_start
        communication_time = comm_end - comm_start
        iteration_times.append(iteration_time)
        communication_times.append(communication_time)
        
        if rank == 0 and step % max(1, num_steps // 10) == 0:
            print(f"Step {step}/{num_steps}: Iter Time: {iteration_time*1000:.2f}ms, "
                  f"Comm Time: {communication_time*1000:.2f}ms, Loss: {loss.item():.4f}")
    
    # Save timing statistics (only rank 0)
    if rank == 0 and result_file is not None:
        timing_stats = {
            'iteration_times': iteration_times,
            'communication_times': communication_times,
            'avg_iteration_time': sum(iteration_times) / len(iteration_times) if iteration_times else 0,
            'avg_communication_time': sum(communication_times) / len(communication_times) if communication_times else 0,
            'total_iteration_time': sum(iteration_times),
            'total_communication_time': sum(communication_times),
            'communication_ratio': sum(communication_times) / sum(iteration_times) if sum(iteration_times) > 0 else 0,
            'ddp_type': ddp_type,
            'world_size': world_size,
            'model_config': {
                'vocab_size': vocab_size,
                'context_length': context_length,
                'd_model': d_model,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'd_ff': d_ff,
                'batch_size': batch_size,
            }
        }
        torch.save(timing_stats, result_file)
    
    cleanup_distributed()


def run_training(
    rank: int,
    world_size: int,
    ddp_type: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    batch_size: int,
    num_steps: int,
    warmup_steps: int,
    backend: str,
    bucket_size_mb: float,
    result_file: Optional[str],
):
    """Wrapper function for multiprocessing spawn."""
    train_step_ddp(
        rank, world_size, ddp_type, vocab_size, context_length,
        d_model, num_layers, num_heads, d_ff, batch_size,
        num_steps, warmup_steps, backend, bucket_size_mb, result_file
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Large model with different DDP implementations"
    )
    
    # Model configuration (Large model)
    parser.add_argument(
        "--vocab_size", type=int, default=10000,
        help="Vocabulary size (default: 10000)"
    )
    parser.add_argument(
        "--context_length", type=int, default=512,
        help="Context length / sequence length (default: 512)"
    )
    parser.add_argument(
        "--d_model", type=int, default=1280,
        help="Model dimension (default: 1280 for Large)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=36,
        help="Number of transformer layers (default: 36 for Large)"
    )
    parser.add_argument(
        "--num_heads", type=int, default=20,
        help="Number of attention heads (default: 20 for Large)"
    )
    parser.add_argument(
        "--d_ff", type=int, default=5120,
        help="Feed-forward dimension (default: 5120 for Large)"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size per process (default: 4)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="Number of training steps to measure (default: 10)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=3,
        help="Number of warmup steps (default: 3)"
    )
    
    # DDP configuration
    parser.add_argument(
        "--ddp_type", type=str, default="naive",
        choices=["naive", "flattened", "overlapped", "bucketed"],
        help="Type of DDP implementation (default: naive)"
    )
    parser.add_argument(
        "--bucket_size_mb", type=float, default=25.0,
        help="Bucket size in MB for bucketed DDP (default: 25.0, only used when ddp_type=bucketed)"
    )
    parser.add_argument(
        "--world_size", type=int, default=2,
        help="Number of processes / GPUs (default: 2)"
    )
    parser.add_argument(
        "--backend", type=str, default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend (default: nccl)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Large Model DDP Benchmarking")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  DDP Type: {args.ddp_type}")
    if args.ddp_type == "bucketed":
        print(f"  Bucket Size: {args.bucket_size_mb} MB")
    print(f"  World Size: {args.world_size}")
    print(f"  Backend: {args.backend}")
    print(f"  Model: Large (d_model={args.d_model}, num_layers={args.num_layers}, "
          f"num_heads={args.num_heads}, d_ff={args.d_ff})")
    
    # Calculate and display model parameters
    vocab_size = args.vocab_size
    d_model = args.d_model
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_ff = args.d_ff
    
    # Calculate model parameters
    embedding_params = vocab_size * d_model
    attn_params_per_layer = 4 * d_model * d_model  # QKV: 3*d_model^2, Out: d_model^2
    ffn_params_per_layer = 3 * d_model * d_ff  # SwiGLU: w1, w2, w3
    ln_params_per_layer = 2 * d_model
    block_params = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    all_blocks_params = num_layers * block_params
    final_ln_params = d_model
    lm_head_params = d_model * vocab_size
    total_params = embedding_params + all_blocks_params + final_ln_params + lm_head_params
    
    print(f"\nModel Parameters:")
    print(f"  Embedding: {embedding_params:,} parameters")
    print(f"  Transformer Blocks ({num_layers} layers): {all_blocks_params:,} parameters")
    print(f"    - Per layer: {block_params:,} parameters")
    print(f"      * Attention: {attn_params_per_layer:,} parameters")
    print(f"      * FFN: {ffn_params_per_layer:,} parameters")
    print(f"      * LayerNorm: {ln_params_per_layer:,} parameters")
    print(f"  Final LayerNorm: {final_ln_params:,} parameters")
    print(f"  LM Head: {lm_head_params:,} parameters")
    print(f"  Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Model Memory (FP32): {total_params * 4 / (1024**3):.2f} GB")
    print(f"  Context Length: {args.context_length}")
    print(f"  Batch Size (per process): {args.batch_size}")
    print(f"  Total Batch Size: {args.batch_size * args.world_size}")
    print(f"  Warmup Steps: {args.warmup_steps}")
    print(f"  Measurement Steps: {args.num_steps}")
    
    # Use temporary file to store results
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        result_file = tmp_file.name
    
    try:
        # Spawn DDP processes
        mp.spawn(
            run_training,
            args=(
                args.world_size, args.ddp_type, args.vocab_size,
                args.context_length, args.d_model, args.num_layers,
                args.num_heads, args.d_ff, args.batch_size,
                args.num_steps, args.warmup_steps, args.backend,
                args.bucket_size_mb, result_file
            ),
            nprocs=args.world_size,
            join=True
        )
        
        # Load and display results
        if os.path.exists(result_file):
            timing_stats = torch.load(result_file)
            
            print(f"\n{'=' * 80}")
            print("Benchmark Results")
            print(f"{'=' * 80}")
            print(f"\nDDP Type: {timing_stats['ddp_type']}")
            print(f"World Size: {timing_stats['world_size']}")
            print(f"\nTiming Statistics:")
            print(f"  Average time per training step: {timing_stats['avg_iteration_time']*1000:.2f} ms")
            print(f"  Average time for gradient communication: {timing_stats['avg_communication_time']*1000:.2f} ms")
            print(f"  Total training time: {timing_stats['total_iteration_time']:.3f} s")
            print(f"  Total communication time: {timing_stats['total_communication_time']:.3f} s")
            print(f"  Communication overhead: {timing_stats['communication_ratio']*100:.2f}%")
            print(f"\nDetailed breakdown:")
            computation_time = timing_stats['avg_iteration_time'] - timing_stats['avg_communication_time']
            print(f"  - Computation time per step: {computation_time*1000:.2f} ms")
            print(f"  - Communication time per step: {timing_stats['avg_communication_time']*1000:.2f} ms")
            if computation_time > 0:
                comm_comp_ratio = timing_stats['avg_communication_time'] / computation_time
                print(f"  - Communication/Computation ratio: {comm_comp_ratio:.2f}x")
            
            print(f"\n{'=' * 80}")
        else:
            print("ERROR: Failed to get timing statistics!")
            
    finally:
        # Clean up temporary file
        if os.path.exists(result_file):
            os.remove(result_file)


if __name__ == "__main__":
    main()

