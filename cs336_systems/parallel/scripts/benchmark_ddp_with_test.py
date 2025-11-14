#!/usr/bin/env python3
"""
Test script for naive DDP implementation.

This script verifies the correctness of the naive DDP implementation by:
1. Training a small toy model on randomly-generated data using single-process training
2. Training the same model using naive DDP with multiple processes
3. Comparing the final weights to ensure they match

Examples:
    # Run from the project root directory
    python -m cs336_systems.parallel.scripts.test_naive_ddp
    
    # Or run directly
    python cs336_systems/parallel/scripts/test_naive_ddp.py
    
    # Or from the parallel directory
    cd cs336_systems/parallel
    python scripts/test_naive_ddp.py
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
from copy import deepcopy

# Import the naive DDP implementation
from cs336_systems.parallel.naive_ddp import NaiveDDP
from cs336_systems.parallel.flattened_ddp import FlattenedDDP
from cs336_systems.parallel.overlapped_ddp import OverlappedDDP
from cs336_systems.parallel.bucketed_ddp import BucketedDDP


class SimpleToyModel(nn.Module):
    """A simple toy model for testing DDP."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup_distributed(rank, world_size, backend="gloo"):
    """Initialize the process group for distributed communication.
    
    Args:
        rank: The rank of the current process (0 to world_size-1)
        world_size: Total number of processes in the group
        backend: Backend to use ("gloo" for CPU, "nccl" for GPU)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"  # Use different port to avoid conflicts
    
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


def train_single_process(model, data, labels, num_epochs=10, lr=0.01):
    """Train a model using single-process training (baseline).
    
    Args:
        model: The model to train
        data: Full training data (batch_size, input_size)
        labels: Full training labels (batch_size, output_size)
        num_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model state dict
    """
    device = next(model.parameters()).device
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    data = data.to(device)
    labels = labels.to(device)
    
    # Warm-up iteration (to match DDP training)
    if num_epochs > 0:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Single-process training - Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Move state dict to CPU to avoid device mismatch issues
    state_dict = model.state_dict()
    cpu_state_dict = {k: v.cpu() if v.is_cuda else v for k, v in state_dict.items()}
    return cpu_state_dict


def train_ddp_process(rank, world_size, model_state_dict, all_data, all_labels, 
                      num_epochs=10, lr=0.01, backend="gloo", result_file=None, 
                      timing_file=None, ddp_type="naive", bucket_size_mb=25.0):
    """Train a model using DDP in a single process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        model_state_dict: Initial model state dict (from rank 0)
        all_data: Full training data
        all_labels: Full training labels
        num_epochs: Number of training epochs
        lr: Learning rate
        backend: Distributed backend
        result_file: Path to file to save final state dict (only rank 0 writes)
        timing_file: Path to file to save timing statistics (only rank 0 writes)
    """
    device = setup_distributed(rank, world_size, backend)
    
    # Create model and load initial state
    model = SimpleToyModel()
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    
    # Wrap with DDP based on type
    if ddp_type == "naive":
        ddp_model = NaiveDDP(model)
    elif ddp_type == "flattened":
        ddp_model = FlattenedDDP(model)
    elif ddp_type == "overlapped":
        ddp_model = OverlappedDDP(model)
    elif ddp_type == "bucketed":
        ddp_model = BucketedDDP(model, bucket_size_mb=bucket_size_mb)
    else:
        raise ValueError(f"Unknown DDP type: {ddp_type}")
    
    # Split data across ranks
    batch_size = all_data.size(0)
    local_batch_size = batch_size // world_size
    offset = rank * local_batch_size
    local_data = all_data[offset:offset + local_batch_size].to(device)
    local_labels = all_labels[offset:offset + local_batch_size].to(device)
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Timing statistics
    iteration_times = []
    communication_times = []
    
    # Warm-up iteration (to match single-process training and warm up CUDA)
    if num_epochs > 0:
        optimizer.zero_grad()
        outputs = ddp_model(local_data)
        loss = criterion(outputs, local_labels)
        loss.backward()
        # Synchronize gradients based on DDP type
        if ddp_type == "overlapped" or ddp_type == "bucketed":
            ddp_model.finish_gradient_synchronization()
        else:
            ddp_model.sync_gradients()
        optimizer.step()
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Actual timing measurements
    for epoch in range(num_epochs):
        # Synchronize before timing (for GPU)
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        iteration_start = time.time()
        
        optimizer.zero_grad()
        outputs = ddp_model(local_data)
        loss = criterion(outputs, local_labels)
        loss.backward()
        
        # Measure communication time
        # For overlapped DDP, communication happens during backward pass
        # We measure the time to finish synchronization (waiting for async ops)
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
        comm_start = time.time()
        
        # Synchronize gradients after backward pass
        # For overlapped and bucketed DDP, this waits for all async all-reduces to complete
        if ddp_type == "overlapped" or ddp_type == "bucketed":
            ddp_model.finish_gradient_synchronization()
        else:
            ddp_model.sync_gradients()
        
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
        
        if rank == 0 and epoch % 5 == 0:
            print(f"DDP training (rank {rank}) - Epoch {epoch}, Loss: {loss.item():.6f}, "
                  f"Iter Time: {iteration_time*1000:.2f}ms, Comm Time: {communication_time*1000:.2f}ms")
    
    # Get final state dict and save to file (only rank 0)
    # Move to CPU before saving to avoid device mismatch issues
    if rank == 0:
        if result_file is not None:
            state_dict = ddp_model.state_dict()
            # Move all tensors to CPU
            cpu_state_dict = {k: v.cpu() if v.is_cuda else v for k, v in state_dict.items()}
            torch.save(cpu_state_dict, result_file)
        
        # Save timing statistics
        if timing_file is not None:
            timing_stats = {
                'iteration_times': iteration_times,
                'communication_times': communication_times,
                'avg_iteration_time': sum(iteration_times) / len(iteration_times) if iteration_times else 0,
                'avg_communication_time': sum(communication_times) / len(communication_times) if communication_times else 0,
                'total_iteration_time': sum(iteration_times),
                'total_communication_time': sum(communication_times),
                'communication_ratio': sum(communication_times) / sum(iteration_times) if sum(iteration_times) > 0 else 0,
            }
            torch.save(timing_stats, timing_file)
    
    cleanup_distributed()


def run_ddp_training(rank, world_size, model_state_dict, all_data, all_labels, 
                     num_epochs, lr, backend, result_file, timing_file, ddp_type, bucket_size_mb):
    """Wrapper function for multiprocessing spawn."""
    train_ddp_process(rank, world_size, model_state_dict, all_data, all_labels,
                     num_epochs, lr, backend, result_file, timing_file, ddp_type, bucket_size_mb)


def compare_models(state_dict1, state_dict2, tolerance=1e-5):
    """Compare two model state dicts to check if they match.
    
    Args:
        state_dict1: First model state dict
        state_dict2: Second model state dict
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if models match, False otherwise
    """
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        print("ERROR: Model state dicts have different keys!")
        return False
    
    all_match = True
    for key in state_dict1.keys():
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        # Move both tensors to CPU for comparison (in case they're on different devices)
        if param1.is_cuda:
            param1 = param1.cpu()
        if param2.is_cuda:
            param2 = param2.cpu()
        
        if not torch.allclose(param1, param2, atol=tolerance, rtol=tolerance):
            print(f"ERROR: Parameter '{key}' does not match!")
            print(f"  Single-process: {param1}")
            print(f"  DDP: {param2}")
            print(f"  Max difference: {(param1 - param2).abs().max().item()}")
            all_match = False
        else:
            max_diff = (param1 - param2).abs().max().item()
            print(f"✓ Parameter '{key}' matches (max diff: {max_diff:.2e})")
    
    return all_match


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Benchmark DDP implementations with a toy model"
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
        "--num_epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=100,
        help="Batch size (default: 100)"
    )
    parser.add_argument(
        "--input_size", type=int, default=10,
        help="Input size of the toy model (default: 10)"
    )
    parser.add_argument(
        "--output_size", type=int, default=5,
        help="Output size of the toy model (default: 5)"
    )
    parser.add_argument(
        "--backend", type=str, default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend (default: nccl)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Testing DDP Implementations with Toy Model")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    world_size = args.world_size
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    input_size = args.input_size
    output_size = args.output_size
    backend = args.backend
    ddp_type = args.ddp_type
    bucket_size_mb = args.bucket_size_mb
    
    print(f"\nConfiguration:")
    print(f"  DDP Type: {ddp_type}")
    if ddp_type == "bucketed":
        print(f"  Bucket Size: {bucket_size_mb} MB")
    print(f"  World size: {world_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Backend: {backend}")
    
    # Generate random data
    print(f"\nGenerating random training data...")
    all_data = torch.randn(batch_size, input_size)
    all_labels = torch.randn(batch_size, output_size)
    
    # Create initial model
    print(f"\nCreating initial model...")
    initial_model = SimpleToyModel(input_size=input_size, hidden_size=20, output_size=output_size)
    initial_state_dict = initial_model.state_dict()
    
    # Train with single process (baseline)
    print(f"\n{'=' * 80}")
    print("Training with single process (baseline)...")
    print(f"{'=' * 80}")
    single_process_model = SimpleToyModel(input_size=input_size, hidden_size=20, output_size=output_size)
    single_process_model.load_state_dict(deepcopy(initial_state_dict))
    single_process_state_dict = train_single_process(
        single_process_model, all_data, all_labels, num_epochs, lr
    )
    
    # Train with DDP
    print(f"\n{'=' * 80}")
    print(f"Training with DDP ({world_size} processes, {ddp_type} mode)...")
    print(f"{'=' * 80}")
    
    # Use temporary files to store results and timing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        result_file = tmp_file.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        timing_file = tmp_file.name
    
    try:
        # Use multiprocessing to spawn DDP processes
        mp.spawn(
            run_ddp_training,
            args=(world_size, deepcopy(initial_state_dict), all_data, all_labels, 
                  num_epochs, lr, backend, result_file, timing_file, ddp_type, bucket_size_mb),
            nprocs=world_size,
            join=True
        )
        
        # Load state dict from file
        if not os.path.exists(result_file):
            print("ERROR: Failed to get DDP model state dict!")
            return
        
        ddp_state_dict = torch.load(result_file)
        
        # Load and display timing statistics
        timing_stats = None
        if os.path.exists(timing_file):
            timing_stats = torch.load(timing_file)
    finally:
        # Clean up temporary files
        if os.path.exists(result_file):
            os.remove(result_file)
        if os.path.exists(timing_file):
            os.remove(timing_file)
    
    # Compare results
    print(f"\n{'=' * 80}")
    print("Comparing Results")
    print(f"{'=' * 80}")
    
    models_match = compare_models(single_process_state_dict, ddp_state_dict)
    
    # Display timing statistics
    if timing_stats is not None:
        print(f"\n{'=' * 80}")
        print("Timing Statistics")
        print(f"{'=' * 80}")
        print(f"Average time per training iteration: {timing_stats['avg_iteration_time']*1000:.2f} ms")
        print(f"Average time for gradient communication: {timing_stats['avg_communication_time']*1000:.2f} ms")
        print(f"Total training time: {timing_stats['total_iteration_time']:.3f} s")
        print(f"Total communication time: {timing_stats['total_communication_time']:.3f} s")
        print(f"Communication overhead: {timing_stats['communication_ratio']*100:.2f}%")
        print(f"\nDetailed breakdown:")
        computation_time = timing_stats['avg_iteration_time'] - timing_stats['avg_communication_time']
        print(f"  - Computation time per iteration: {computation_time*1000:.2f} ms")
        print(f"  - Communication time per iteration: {timing_stats['avg_communication_time']*1000:.2f} ms")
        if computation_time > 0:
            comm_comp_ratio = timing_stats['avg_communication_time'] / computation_time
            print(f"  - Communication/Computation ratio: {comm_comp_ratio:.2f}x")
        
        # Add DDP type and bucket size to timing stats for easier debugging/logging
        timing_stats['ddp_type'] = ddp_type
        timing_stats['bucket_size_mb'] = bucket_size_mb
        # Save timing stats with ddp_type and bucket_size_mb for easy processing
        torch.save(timing_stats, timing_file)

    print(f"\n{'=' * 80}")
    if models_match:
        print("✓ SUCCESS: DDP training produces identical weights to single-process training!")
        print("  The DDP implementation is correct.")
    else:
        print("✗ FAILURE: DDP training produces different weights!")
        print("  The DDP implementation may have bugs.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

