#!/usr/bin/env python3
"""
Flattened DDP implementation that batches all-reduce operations.

This module provides a DDP wrapper that:
1. Broadcasts model parameters from rank 0 to all other ranks at initialization
2. Synchronizes gradients across all ranks using a single batched all-reduce call
   by flattening all gradients into a single tensor
3. Averages gradients across all ranks before optimizer step
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class FlattenedDDP(nn.Module):
    """
    A DDP implementation that uses a single batched all-reduce for gradient synchronization.
    
    This wrapper handles:
    - Parameter broadcasting from rank 0 to all ranks at initialization
    - Gradient synchronization via a single all-reduce on flattened gradients
    
    Usage:
        model = MyModel()
        ddp_model = FlattenedDDP(model)
        # ... training loop ...
        loss.backward()
        ddp_model.sync_gradients()  # Call after backward, before optimizer.step()
        optimizer.step()
    """
    
    def __init__(self, module: nn.Module):
        """
        Initialize the FlattenedDDP wrapper.
        
        Args:
            module: The underlying model to wrap with DDP.
        """
        super().__init__()
        self.module = module
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
    
    def _broadcast_parameters(self):
        """
        Broadcast all parameters from rank 0 to all other ranks.
        This ensures all ranks start with identical model parameters.
        """
        for param in self.module.parameters():
            # Broadcast parameter from rank 0 to all ranks
            dist.broadcast(param.data, src=0)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the underlying module.
        
        Args:
            *args: Positional arguments to pass to the module
            **kwargs: Keyword arguments to pass to the module
            
        Returns:
            Output from the module's forward pass
        """
        return self.module(*args, **kwargs)
    
    def sync_gradients(self):
        """
        Synchronize gradients across all ranks using a single batched all-reduce.
        
        This method should be called after backward() but before optimizer.step().
        It averages gradients across all ranks by:
        1. Collecting all gradients that require grad
        2. Flattening them into a single tensor
        3. Performing a single all-reduce on the flattened tensor
        4. Unflattening the result back to individual parameter gradients
        5. Dividing by world_size to get the average
        
        Only parameters with requires_grad=True and grad is not None are synchronized.
        """
        world_size = dist.get_world_size()
        
        # Collect all gradients that need to be synchronized
        grads_to_sync = []
        param_refs = []  # Keep references to original parameters for unflattening
        
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                grads_to_sync.append(param.grad.data)
                param_refs.append(param)
        
        if len(grads_to_sync) == 0:
            # No gradients to synchronize
            return
        
        # Flatten all gradients into a single tensor
        flat_grads = _flatten_dense_tensors(grads_to_sync)
        
        # Perform all-reduce on the flattened tensor
        # op=dist.ReduceOp.SUM sums gradients from all ranks
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        
        # Divide by world_size to get the average gradient
        flat_grads /= world_size
        
        # Unflatten the averaged gradients back to individual parameter gradients
        unflattened_grads = _unflatten_dense_tensors(flat_grads, grads_to_sync)
        
        # Copy the averaged gradients back to the original parameter gradients
        for param, avg_grad in zip(param_refs, unflattened_grads):
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


def create_flattened_ddp_model(module: nn.Module) -> FlattenedDDP:
    """
    Convenience function to create a FlattenedDDP wrapper around a module.
    
    Args:
        module: The model to wrap with DDP.
        
    Returns:
        A FlattenedDDP instance wrapping the module.
    """
    return FlattenedDDP(module)

