#!/usr/bin/env python3
"""
Overlapped DDP implementation that overlaps computation with communication.

This module provides a DDP wrapper that:
1. Broadcasts model parameters from rank 0 to all other ranks at initialization
2. Asynchronously all-reduces individual parameter gradients as soon as they're ready
   during the backward pass, overlapping computation with communication
3. Provides a method to wait for all communication to complete before optimizer step
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional


class OverlappedDDP(nn.Module):
    """
    A DDP implementation that overlaps computation with communication by asynchronously
    all-reducing individual parameter gradients as soon as they're ready during backward pass.
    
    This wrapper handles:
    - Parameter broadcasting from rank 0 to all ranks at initialization
    - Asynchronous gradient synchronization via all-reduce hooks during backward pass
    
    Usage:
        model = MyModel()
        ddp_model = OverlappedDDP(model)
        # ... training loop ...
        loss.backward()  # Gradients are automatically all-reduced as they become ready
        ddp_model.finish_gradient_synchronization()  # Wait for all communication to complete
        optimizer.step()
    """
    
    def __init__(self, module: nn.Module):
        """
        Initialize the OverlappedDDP wrapper.
        
        Args:
            module: The underlying model to wrap with DDP.
        """
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        
        # List to store communication handles and parameter references for asynchronous all-reduce operations
        # Each element is a tuple: (handle, param)
        self._communication_handles: List[tuple] = []
        
        # Set to track which parameters have already triggered all-reduce
        # This prevents multiple all-reduces if the hook is called multiple times
        self._params_with_pending_allreduce: set = set()
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Register backward hooks on all parameters to trigger async all-reduce
        # when gradients become ready
        self._register_gradient_hooks()
    
    def _broadcast_parameters(self):
        """
        Broadcast all parameters from rank 0 to all other ranks.
        This ensures all ranks start with identical model parameters.
        """
        for param in self.module.parameters():
            # Broadcast parameter from rank 0 to all ranks
            dist.broadcast(param.data, src=0)
    
    def _register_gradient_hooks(self):
        """
        Register post-accumulate gradient hooks on all parameters.
        These hooks will be called automatically when each parameter's gradient
        is ready during the backward pass, triggering an asynchronous all-reduce.
        """
        for param in self.module.parameters():
            if param.requires_grad:
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
            This function marks the parameter for async all-reduce.
            
            IMPORTANT: We cannot modify param.grad.data directly in the hook because:
            1. The hook is called during backward pass, and modifying gradients at this point
               can interfere with ongoing gradient computations
            2. We need to ensure the gradient is fully accumulated before all-reduce
            
            Instead, we mark the parameter and will perform all-reduce after backward completes.
            However, to achieve overlap, we still want to start all-reduce as soon as possible.
            
            The solution: Use param.grad.data (not grad) and ensure we're working with
            the actual gradient storage, not a temporary view.
            
            Args:
                grad: The gradient tensor (may be a view of param.grad)
                
            Returns:
                The gradient tensor (unchanged)
            """
            if grad is not None and param.grad is not None:
                # Prevent multiple all-reduces for the same parameter
                # The hook might be called multiple times during backward pass
                if param not in self._params_with_pending_allreduce:
                    # Mark this parameter as having a pending all-reduce
                    self._params_with_pending_allreduce.add(param)
                    
                    # CRITICAL: We must use param.grad.data, not grad
                    # grad might be a temporary view that gets invalidated
                    # param.grad.data is the actual gradient storage
                    # 
                    # Asynchronously all-reduce the gradient
                    # async_op=True returns immediately without waiting
                    # This allows computation of other gradients to continue while communication happens
                    handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                    
                    # Store the handle so we can wait for it later
                    # We also need to store the parameter reference to divide by world_size later
                    self._communication_handles.append((handle, param))
            # Return grad unchanged
            return grad
        
        return hook
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the underlying module.
        
        Args:
            *args: Positional arguments to pass to the module
            **kwargs: Keyword arguments to pass to the module
            
        Returns:
            Output from the module's forward pass
        """
        # Clear communication handles and pending all-reduce set at the start of each forward pass
        # (in case finish_gradient_synchronization wasn't called)
        self._communication_handles.clear()
        self._params_with_pending_allreduce.clear()
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous all-reduce operations to complete.
        
        This method should be called after backward() but before optimizer.step().
        It ensures that:
        1. All communication operations are queued on GPU
        2. All gradients are averaged across all ranks
        3. The gradients are ready to be used by the optimizer
        """
        # Wait for all asynchronous all-reduce operations to be queued
        # and then divide by world_size to get the average gradient
        for handle, param in self._communication_handles:
            # Wait for the all-reduce to be queued on GPU
            handle.wait()
            
            # Now that all-reduce has completed (summed), divide by world_size
            # to get the average gradient
            if param.requires_grad and param.grad is not None:
                param.grad.data /= self.world_size
        
        # Clear the handles list and pending all-reduce set for the next iteration
        self._communication_handles.clear()
        self._params_with_pending_allreduce.clear()
    
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


def create_overlapped_ddp_model(module: nn.Module) -> OverlappedDDP:
    """
    Convenience function to create an OverlappedDDP wrapper around a module.
    
    Args:
        module: The model to wrap with DDP.
        
    Returns:
        An OverlappedDDP instance wrapping the module.
    """
    return OverlappedDDP(module)

