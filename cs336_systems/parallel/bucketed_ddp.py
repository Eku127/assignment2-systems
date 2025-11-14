#!/usr/bin/env python3
"""
Bucketed DDP implementation that overlaps computation with communication using gradient bucketing.

This module provides a DDP wrapper that:
1. Broadcasts model parameters from rank 0 to all other ranks at initialization
2. Organizes parameters into buckets based on bucket_size_mb
3. Asynchronously all-reduces bucket gradients as soon as all parameters in a bucket are ready
   during the backward pass, overlapping computation with communication
4. Provides a method to wait for all communication to complete before optimizer step
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from typing import List, Dict, Set, Optional


class BucketedDDP(nn.Module):
    """
    A DDP implementation that overlaps computation with communication by asynchronously
    all-reducing buckets of parameter gradients as soon as all parameters in a bucket are ready.
    
    This wrapper handles:
    - Parameter broadcasting from rank 0 to all ranks at initialization
    - Organizing parameters into buckets based on bucket_size_mb
    - Asynchronous gradient synchronization via all-reduce hooks during backward pass
    
    Usage:
        model = MyModel()
        ddp_model = BucketedDDP(model, bucket_size_mb=25.0)
        # ... training loop ...
        loss.backward()  # Gradients are automatically all-reduced as buckets become ready
        ddp_model.finish_gradient_synchronization()  # Wait for all communication to complete
        optimizer.step()
    """
    
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        """
        Initialize the BucketedDDP wrapper.
        
        Args:
            module: The underlying model to wrap with DDP.
            bucket_size_mb: Maximum size of each bucket in megabytes.
        """
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)  # Convert MB to bytes
        
        # List to store communication handles for asynchronous bucket all-reduce operations
        # Each element is a tuple: (handle, bucket_id, bucket_params)
        self._communication_handles: List[tuple] = []
        
        # Dictionary mapping bucket_id to list of parameters in that bucket
        self._bucket_params: Dict[int, List[nn.Parameter]] = {}
        
        # Dictionary mapping parameter to its bucket_id
        self._param_to_bucket: Dict[nn.Parameter, int] = {}
        
        # Dictionary mapping bucket_id to set of parameters that have gradients ready
        self._bucket_ready_params: Dict[int, Set[nn.Parameter]] = {}
        
        # Set to track which parameters have already triggered their hook
        self._params_with_hook_called: Set[nn.Parameter] = set()
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Organize parameters into buckets (in reverse order, as gradients become ready in reverse)
        self._organize_parameters_into_buckets()
        
        # Register backward hooks on all parameters to trigger async bucket all-reduce
        # when all parameters in a bucket have gradients ready
        self._register_gradient_hooks()
    
    def _broadcast_parameters(self):
        """
        Broadcast all parameters from rank 0 to all other ranks.
        This ensures all ranks start with identical model parameters.
        """
        for param in self.module.parameters():
            # Broadcast parameter from rank 0 to all ranks
            dist.broadcast(param.data, src=0)
    
    def _get_parameter_size_bytes(self, param: nn.Parameter) -> int:
        """
        Calculate the size of a parameter in bytes.
        
        Args:
            param: The parameter tensor
            
        Returns:
            Size in bytes
        """
        if param.numel() == 0:
            return 0
        # Size = number of elements * bytes per element
        return param.numel() * param.element_size()
    
    def _organize_parameters_into_buckets(self):
        """
        Organize parameters into buckets based on bucket_size_mb.
        Parameters are allocated in reverse order (as gradients become ready in reverse during backward).
        """
        # Get all parameters that require grad in reverse order
        # (gradients become ready in reverse order during backward pass)
        all_params = list(self.module.parameters())
        params_with_grad = [p for p in reversed(all_params) if p.requires_grad]
        
        if len(params_with_grad) == 0:
            return
        
        bucket_id = 0
        current_bucket_size = 0
        current_bucket_params = []
        
        for param in params_with_grad:
            param_size = self._get_parameter_size_bytes(param)
            
            # If adding this parameter would exceed the bucket size, start a new bucket
            if current_bucket_size + param_size > self.bucket_size_bytes and len(current_bucket_params) > 0:
                # Save current bucket
                self._bucket_params[bucket_id] = current_bucket_params
                for p in current_bucket_params:
                    self._param_to_bucket[p] = bucket_id
                self._bucket_ready_params[bucket_id] = set()
                
                # Start new bucket
                bucket_id += 1
                current_bucket_size = 0
                current_bucket_params = []
            
            # Add parameter to current bucket
            current_bucket_params.append(param)
            current_bucket_size += param_size
        
        # Don't forget the last bucket
        if len(current_bucket_params) > 0:
            self._bucket_params[bucket_id] = current_bucket_params
            for p in current_bucket_params:
                self._param_to_bucket[p] = bucket_id
            self._bucket_ready_params[bucket_id] = set()
    
    def _register_gradient_hooks(self):
        """
        Register post-accumulate gradient hooks on all parameters.
        These hooks will be called automatically when each parameter's gradient
        is ready during the backward pass, checking if the bucket is ready for all-reduce.
        """
        for param in self.module.parameters():
            if param.requires_grad and param in self._param_to_bucket:
                # Register a hook that will be called when the gradient is accumulated
                param.register_post_accumulate_grad_hook(self._make_gradient_hook(param))
    
    def _make_gradient_hook(self, param: nn.Parameter):
        """
        Create a gradient hook function for a specific parameter.
        
        Args:
            param: The parameter to create a hook for
            
        Returns:
            A hook function that will be called when the gradient is ready
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            """
            Hook function called when the parameter's gradient is ready.
            This function checks if the bucket is ready and triggers async all-reduce if so.
            
            Args:
                grad: The gradient tensor (may be a view of param.grad)
                
            Returns:
                The gradient tensor (unchanged)
            """
            if grad is not None and param.grad is not None:
                # Prevent multiple hook calls for the same parameter
                if param not in self._params_with_hook_called:
                    self._params_with_hook_called.add(param)
                    
                    # Get the bucket ID for this parameter
                    bucket_id = self._param_to_bucket.get(param)
                    if bucket_id is None:
                        return grad
                    
                    # Mark this parameter as ready in its bucket
                    self._bucket_ready_params[bucket_id].add(param)
                    
                    # Check if all parameters in this bucket are ready
                    bucket_params = self._bucket_params[bucket_id]
                    if len(self._bucket_ready_params[bucket_id]) == len(bucket_params):
                        # All parameters in the bucket are ready, trigger async all-reduce
                        self._all_reduce_bucket(bucket_id, bucket_params)
            
            return grad
        
        return hook
    
    def _all_reduce_bucket(self, bucket_id: int, bucket_params: List[nn.Parameter]):
        """
        Perform asynchronous all-reduce on a bucket of gradients.
        
        Args:
            bucket_id: The ID of the bucket
            bucket_params: List of parameters in the bucket
        """
        # Collect gradients for all parameters in the bucket
        grads_to_sync = []
        param_refs = []
        
        for param in bucket_params:
            if param.requires_grad and param.grad is not None:
                grads_to_sync.append(param.grad.data)
                param_refs.append(param)
        
        if len(grads_to_sync) == 0:
            return
        
        # Flatten all gradients in the bucket into a single tensor
        flat_grads = _flatten_dense_tensors(grads_to_sync)
        
        # Asynchronously all-reduce the flattened gradients
        # async_op=True returns immediately without waiting
        handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
        
        # Store the handle along with bucket info for later processing
        self._communication_handles.append((handle, bucket_id, param_refs, flat_grads, grads_to_sync))
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the underlying module.
        
        Args:
            *args: Positional arguments to pass to the module
            **kwargs: Keyword arguments to pass to the module
            
        Returns:
            Output from the module's forward pass
        """
        # Clear communication handles and bucket ready tracking at the start of each forward pass
        # (in case finish_gradient_synchronization wasn't called)
        self._communication_handles.clear()
        self._params_with_hook_called.clear()
        for bucket_id in self._bucket_ready_params:
            self._bucket_ready_params[bucket_id].clear()
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous bucket all-reduce operations to complete.
        
        This method should be called after backward() but before optimizer.step().
        It ensures that:
        1. All communication operations are queued on GPU
        2. All gradients are averaged across all ranks
        3. The gradients are ready to be used by the optimizer
        
        This method also handles buckets that weren't all-reduced during backward
        (e.g., if some parameters in the bucket didn't have gradients because they
        weren't used in the forward pass).
        """
        # Track which buckets have been all-reduced
        all_reduced_bucket_ids = set()
        
        # Wait for all asynchronous bucket all-reduce operations to complete
        for handle, bucket_id, param_refs, flat_grads, grads_to_sync in self._communication_handles:
            # Wait for the all-reduce to be queued on GPU
            handle.wait()
            
            # Now that all-reduce has completed (summed), divide by world_size to get the average
            flat_grads /= self.world_size
            
            # Unflatten the averaged gradients back to individual parameter gradients
            unflattened_grads = _unflatten_dense_tensors(flat_grads, grads_to_sync)
            
            # Copy the averaged gradients back to the parameter gradients
            for param, avg_grad in zip(param_refs, unflattened_grads):
                if param.requires_grad and param.grad is not None:
                    param.grad.data.copy_(avg_grad)
            
            all_reduced_bucket_ids.add(bucket_id)
        
        # Handle buckets that weren't all-reduced during backward
        # This can happen if some parameters in a bucket didn't get gradients
        # (e.g., they weren't used in the forward pass due to conditional logic)
        for bucket_id, bucket_params in self._bucket_params.items():
            if bucket_id not in all_reduced_bucket_ids:
                # Check if any parameters in this bucket have gradients
                params_with_grad = [p for p in bucket_params 
                                   if p.requires_grad and p.grad is not None]
                
                if len(params_with_grad) > 0:
                    # Some parameters have gradients, so we need to all-reduce them
                    # This is a synchronous call since we're already in the sync phase
                    self._all_reduce_bucket_sync(bucket_id, params_with_grad)
        
        # Clear the handles list and bucket ready tracking for the next iteration
        self._communication_handles.clear()
        self._params_with_hook_called.clear()
        for bucket_id in self._bucket_ready_params:
            self._bucket_ready_params[bucket_id].clear()
    
    def _all_reduce_bucket_sync(self, bucket_id: int, bucket_params: List[nn.Parameter]):
        """
        Perform synchronous all-reduce on a bucket of gradients.
        This is used in finish_gradient_synchronization for buckets that weren't
        all-reduced during backward.
        
        Args:
            bucket_id: The ID of the bucket
            bucket_params: List of parameters in the bucket that have gradients
        """
        # Collect gradients for all parameters in the bucket
        grads_to_sync = []
        param_refs = []
        
        for param in bucket_params:
            if param.requires_grad and param.grad is not None:
                grads_to_sync.append(param.grad.data)
                param_refs.append(param)
        
        if len(grads_to_sync) == 0:
            return
        
        # Flatten all gradients in the bucket into a single tensor
        flat_grads = _flatten_dense_tensors(grads_to_sync)
        
        # Synchronously all-reduce the flattened gradients
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)
        
        # Divide by world_size to get the average
        flat_grads /= self.world_size
        
        # Unflatten the averaged gradients back to individual parameter gradients
        unflattened_grads = _unflatten_dense_tensors(flat_grads, grads_to_sync)
        
        # Copy the averaged gradients back to the parameter gradients
        for param, avg_grad in zip(param_refs, unflattened_grads):
            if param.requires_grad and param.grad is not None:
                param.grad.data.copy_(avg_grad)
    
    def state_dict(self):
        """Return the state dict of the underlying module."""
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into the underlying module."""
        return self.module.load_state_dict(state_dict)
    
    def named_parameters(self, prefix='', recurse=True):
        """Return named parameters of the underlying module."""
        return self.module.named_parameters(prefix=prefix, recurse=recurse)
    
    def parameters(self, recurse=True):
        """Return parameters of the underlying module."""
        return self.module.parameters(recurse=recurse)


def create_bucketed_ddp_model(module: nn.Module, bucket_size_mb: float) -> BucketedDDP:
    """
    Convenience function to create a BucketedDDP wrapper around a module.
    
    Args:
        module: The model to wrap with DDP.
        bucket_size_mb: Maximum size of each bucket in megabytes.
        
    Returns:
        A BucketedDDP instance wrapping the module.
    """
    return BucketedDDP(module, bucket_size_mb)

