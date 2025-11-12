#!/usr/bin/env python3
"""
Benchmarking script for Transformer model forward and backward passes.

This script performs basic end-to-end benchmarking of the forward and backward
passes in a Transformer model, with support for warm-up steps and timing measurements.
"""

import argparse
import timeit
import statistics
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM


def generate_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a random batch of token IDs.

    Args:
        batch_size: The batch size.
        context_length: The sequence length.
        vocab_size: The vocabulary size (used to generate valid token IDs).
        device: The device to create the tensor on.

    Returns:
        A tensor of shape (batch_size, context_length) with random token IDs.
    """
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)


def benchmark_model(
    model: nn.Module,
    batch: torch.Tensor,
    warmup_steps: int,
    num_steps: int,
    forward_only: bool = False,
) -> tuple[list[float], list[float]]:
    """Benchmark the model's forward and backward passes.

    Args:
        model: The model to benchmark.
        batch: The input batch tensor.
        warmup_steps: Number of warm-up steps before timing.
        num_steps: Number of steps to time.
        forward_only: If True, only run forward pass. If False, run forward + backward.

    Returns:
        A tuple of (forward_times, backward_times) lists. backward_times will be empty
        if forward_only is True.
    """
    forward_times = []
    backward_times = []

    # Warm-up steps (marked with NVTX so we can filter them out in nsys)
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            if forward_only:
                _ = model(batch)
            else:
                model.zero_grad()
                output = model(batch)
                # Create a dummy loss (sum of logits)
                loss = output.sum()
                loss.backward()
            torch.cuda.synchronize()

    # Timing steps (marked with NVTX for profiling)
    for step in range(num_steps):
        with nvtx.range(f"step_{step}"):
            if forward_only:
                # Forward pass timing only
                torch.cuda.synchronize()  # Ensure previous operations are done
                start_time = timeit.default_timer()
                with nvtx.range("forward_pass"):
                    _ = model(batch)
                torch.cuda.synchronize()  # Wait for forward pass to complete
                forward_time = timeit.default_timer() - start_time
                forward_times.append(forward_time)
            else:
                # Forward pass timing
                model.zero_grad()  # Clear gradients
                torch.cuda.synchronize()  # Ensure previous operations are done
                start_time = timeit.default_timer()
                with nvtx.range("forward_pass"):
                    output = model(batch)
                torch.cuda.synchronize()  # Wait for forward pass to complete
                forward_time = timeit.default_timer() - start_time
                forward_times.append(forward_time)

                # Backward pass timing
                torch.cuda.synchronize()  # Ensure forward pass is done
                start_time = timeit.default_timer()
                with nvtx.range("backward_pass"):
                    loss = output.sum()
                    loss.backward()
                torch.cuda.synchronize()  # Wait for backward pass to complete
                backward_time = timeit.default_timer() - start_time
                backward_times.append(backward_time)

    return forward_times, backward_times


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer model forward and backward passes"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Vocabulary size (default: 10000)",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        required=True,
        help="Context length (sequence length)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=True,
        help="Model dimension",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        required=True,
        help="Number of Transformer layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        required=True,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        required=True,
        help="Feed-forward dimension",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=10000.0,
        help="RoPE theta parameter (default: 10000.0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps (default: 5)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of measurement steps (default: 10)",
    )
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Only run forward pass (no backward pass)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available, falling back to CPU")
        args.device = "cpu"

    # Initialize model
    print("Initializing model...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model = model.to(args.device)
    model.train()  # Set to training mode for backward pass

    # Generate random batch
    print("Generating random batch...")
    batch = generate_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=args.device,
    )

    # Run benchmark
    print(f"Running benchmark (warmup: {args.warmup_steps}, steps: {args.num_steps})...")
    forward_times, backward_times = benchmark_model(
        model=model,
        batch=batch,
        warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        forward_only=args.forward_only,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Forward pass:")
    print(f"  Mean: {statistics.mean(forward_times) * 1000:.3f} ms")
    print(f"  Std:  {statistics.stdev(forward_times) * 1000:.3f} ms")
    print(f"  Min:  {min(forward_times) * 1000:.3f} ms")
    print(f"  Max:  {max(forward_times) * 1000:.3f} ms")

    if not args.forward_only:
        print(f"\nBackward pass:")
        print(f"  Mean: {statistics.mean(backward_times) * 1000:.3f} ms")
        print(f"  Std:  {statistics.stdev(backward_times) * 1000:.3f} ms")
        print(f"  Min:  {min(backward_times) * 1000:.3f} ms")
        print(f"  Max:  {max(backward_times) * 1000:.3f} ms")

        total_times = [f + b for f, b in zip(forward_times, backward_times)]
        print(f"\nTotal (forward + backward):")
        print(f"  Mean: {statistics.mean(total_times) * 1000:.3f} ms")
        print(f"  Std:  {statistics.stdev(total_times) * 1000:.3f} ms")

    print("=" * 60)


if __name__ == "__main__":
    main()

