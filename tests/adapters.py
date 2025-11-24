from __future__ import annotations

from typing import Type

import torch



def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    raise NotImplementedError


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    raise NotImplementedError


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    from cs336_systems.parallel.overlapped_ddp import OverlappedDDP
    return OverlappedDDP(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # Call finish_gradient_synchronization to wait for all async all-reduce operations
    # to complete before the optimizer step
    if hasattr(ddp_model, 'finish_gradient_synchronization'):
        ddp_model.finish_gradient_synchronization()
    else:
        raise AttributeError(
            "DDP model does not have finish_gradient_synchronization method. "
            "Make sure you're using OverlappedDDP or a compatible DDP implementation."
        )


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    from cs336_systems.parallel.bucketed_ddp import BucketedDDP
    return BucketedDDP(module, bucket_size_mb=bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # Call finish_gradient_synchronization to wait for all async bucket all-reduce operations
    # to complete before the optimizer step
    if hasattr(ddp_model, 'finish_gradient_synchronization'):
        ddp_model.finish_gradient_synchronization()
    else:
        raise AttributeError(
            "DDP model does not have finish_gradient_synchronization method. "
            "Make sure you're using BucketedDDP or a compatible DDP implementation."
        )


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    
    Note:
        This function is optional. The BucketedDDP implementation automatically
        clears communication handles and bucket state in the forward() method,
        so this function can be a no-op. However, we keep it for consistency
        with the test interface and potential future use cases.
    """
    # The BucketedDDP.forward() method already clears communication handles
    # and bucket ready tracking at the start of each forward pass.
    # This function is kept for interface consistency but is essentially a no-op
    # since the state will be cleared when forward() is called.
    # If needed in the future, we could add explicit state clearing here.
    pass


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    from cs336_systems.parallel.sharded_optimizer import ShardedOptimizer
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
