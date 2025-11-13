#!/usr/bin/env python3
"""
Benchmark script for all-reduce operation in distributed PyTorch.

This script benchmarks the runtime of all-reduce operations with various configurations:
- Backend: Gloo (CPU) or NCCL (GPU)
- Data sizes: 1MB, 10MB, 100MB, 1GB (float32)
- Number of processes: 2, 4, or 6
"""

import os
import timeit
import statistics
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend="gloo"):
    """Initialize the process group for distributed communication.
    
    Args:
        rank: The rank of the current process (0 to world_size-1)
        world_size: Total number of processes in the group
        backend: Backend to use ("gloo" for CPU, "nccl" for GPU)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # For NCCL backend, set each process to use a different GPU
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but CUDA is not available")
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"NCCL backend requires at least {world_size} GPUs, "
                f"but only {torch.cuda.device_count()} GPUs are available"
            )
        # Set each process to use a different GPU
        torch.cuda.set_device(rank)
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return backend


def benchmark_all_reduce(
    rank,
    world_size,
    backend,
    data_size_mb,
    num_warmup=5,
    num_iterations=10,
):
    """Benchmark all-reduce operation.
    
    Args:
        rank: The rank of the current process
        world_size: Total number of processes
        backend: Backend to use ("gloo" or "nccl")
        data_size_mb: Data size in MB
        num_warmup: Number of warm-up iterations
        num_iterations: Number of measurement iterations
    
    Returns:
        List of timings from this rank
    """
    backend_used = setup(rank, world_size, backend)
    
    # Determine device based on backend
    if backend_used == "nccl":
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    
    # Calculate tensor size (float32 = 4 bytes per element)
    # data_size_mb * 1024 * 1024 / 4 = number of elements
    num_elements = int(data_size_mb * 1024 * 1024 / 4)
    
    # Create random float32 tensor
    data = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    # Warm-up steps
    for _ in range(num_warmup):
        if backend_used == "nccl":
            torch.cuda.synchronize()
        dist.all_reduce(data, async_op=False)
        if backend_used == "nccl":
            torch.cuda.synchronize()
    
    # Benchmark iterations
    timings = []
    for _ in range(num_iterations):
        if backend_used == "nccl":
            torch.cuda.synchronize()  # Ensure previous operations are done
        
        start_time = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        
        if backend_used == "nccl":
            torch.cuda.synchronize()  # Wait for communication to complete
        
        elapsed_time = timeit.default_timer() - start_time
        timings.append(elapsed_time)
    
    # Collect timings from all ranks
    # Create a tensor to hold all timings
    timings_tensor = torch.tensor(timings, dtype=torch.float32, device=device)
    
    # Gather all timings from all ranks
    # all_gather requires a list of tensors (one for each rank)
    gathered_timings = [torch.zeros_like(timings_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_timings, timings_tensor)
    
    # Convert to list of lists (each inner list is timings from one rank)
    all_timings = [t.cpu().tolist() for t in gathered_timings]
    
    # Cleanup
    dist.destroy_process_group()
    
    # All ranks have the same data after all_gather, but only rank 0 will save it
    return all_timings


def benchmark_with_result_collection(rank, world_size, backend, data_size_mb, num_warmup, num_iterations, result_file):
    """Wrapper to collect results from benchmark."""
    all_timings = benchmark_all_reduce(rank, world_size, backend, data_size_mb, num_warmup, num_iterations)
    # Only rank 0 saves results (all ranks have the same data after all_gather)
    if rank == 0:
        import pickle
        with open(result_file, 'wb') as f:
            pickle.dump(all_timings, f)


def run_benchmark(backend, world_size, data_size_mb, num_warmup=5, num_iterations=10):
    """Run benchmark and return aggregated results.
    
    Args:
        backend: Backend to use ("gloo" or "nccl")
        world_size: Number of processes
        data_size_mb: Data size in MB
        num_warmup: Number of warm-up iterations
        num_iterations: Number of measurement iterations
    
    Returns:
        Dictionary with statistics
    """
    # Check prerequisites
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but CUDA is not available")
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"NCCL backend requires at least {world_size} GPUs, "
                f"but only {torch.cuda.device_count()} GPUs are available"
            )
    
    # Use a temporary file to collect results
    import tempfile
    import pickle
    
    result_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    result_file.close()
    
    # Run benchmark
    mp.spawn(
        fn=benchmark_with_result_collection,
        args=(world_size, backend, data_size_mb, num_warmup, num_iterations, result_file.name),
        nprocs=world_size,
        join=True,
    )
    
    # Load results
    try:
        with open(result_file.name, 'rb') as f:
            all_timings = pickle.load(f)
        os.unlink(result_file.name)
    except FileNotFoundError:
        raise RuntimeError("Failed to collect benchmark results")
    
    # Flatten all timings from all ranks
    all_times = [t for rank_timings in all_timings for t in rank_timings]
    
    # Calculate statistics
    return {
        'mean': statistics.mean(all_times) * 1000,  # Convert to ms
        'std': statistics.stdev(all_times) * 1000,
        'min': min(all_times) * 1000,
        'max': max(all_times) * 1000,
        'median': statistics.median(all_times) * 1000,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark all-reduce operation in distributed PyTorch"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Backend to use: 'gloo' for CPU, 'nccl' for GPU (default: gloo)",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        choices=[2, 4, 6],
        help="Number of processes (default: 2, choices: 2, 4, 6)",
    )
    parser.add_argument(
        "--data_size_mb",
        type=float,
        default=1.0,
        help="Data size in MB (default: 1.0, typical values: 1.0, 10.0, 100.0, 1000.0)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warm-up iterations (default: 5)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of measurement iterations (default: 10)",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all combinations of settings and output a table",
    )
    
    args = parser.parse_args()
    
    if args.run_all:
        # Run all combinations
        backends = ["gloo", "nccl"]
        world_sizes = [2, 4, 6]
        data_sizes = [1.0, 10.0, 100.0, 1000.0]  # 1GB = 1000MB
        
        results = []
        
        for backend in backends:
            # Skip NCCL if CUDA is not available
            if backend == "nccl" and not torch.cuda.is_available():
                print(f"Skipping {backend} (CUDA not available)")
                continue
            
            for world_size in world_sizes:
                # Check GPU availability for NCCL
                if backend == "nccl" and torch.cuda.device_count() < world_size:
                    print(f"Skipping {backend} with {world_size} processes (not enough GPUs)")
                    continue
                
                for data_size_mb in data_sizes:
                    print(f"Running: {backend}, {world_size} processes, {data_size_mb}MB...")
                    try:
                        stats = run_benchmark(
                            backend=backend,
                            world_size=world_size,
                            data_size_mb=data_size_mb,
                            num_warmup=args.num_warmup,
                            num_iterations=args.num_iterations,
                        )
                        results.append({
                            'backend': backend,
                            'world_size': world_size,
                            'data_size_mb': data_size_mb,
                            **stats
                        })
                        print(f"  Mean: {stats['mean']:.3f} ms, Std: {stats['std']:.3f} ms")
                    except Exception as e:
                        print(f"  Error: {e}")
                        continue
        
        # Print results table
        print("\n" + "=" * 100)
        print("Benchmark Results")
        print("=" * 100)
        print(f"{'Backend':<10} {'World Size':<12} {'Data Size (MB)':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 100)
        
        for r in results:
            print(
                f"{r['backend']:<10} {r['world_size']:<12} {r['data_size_mb']:<15.1f} "
                f"{r['mean']:<12.3f} {r['std']:<12.3f} {r['min']:<12.3f} {r['max']:<12.3f}"
            )
        
        print("=" * 100)
        
    else:
        # Run single benchmark
        print(f"Running benchmark: {args.backend}, {args.world_size} processes, {args.data_size_mb}MB")
        stats = run_benchmark(
            backend=args.backend,
            world_size=args.world_size,
            data_size_mb=args.data_size_mb,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
        )
        
        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"Backend: {args.backend}")
        print(f"World Size: {args.world_size}")
        print(f"Data Size: {args.data_size_mb} MB")
        print(f"\nTiming Statistics (across all ranks):")
        print(f"  Mean:   {stats['mean']:.3f} ms")
        print(f"  Std:    {stats['std']:.3f} ms")
        print(f"  Min:    {stats['min']:.3f} ms")
        print(f"  Max:    {stats['max']:.3f} ms")
        print(f"  Median: {stats['median']:.3f} ms")
        print("=" * 60)


if __name__ == "__main__":
    main()
