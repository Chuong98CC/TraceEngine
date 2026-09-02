"""Small nn.Module helpers used by the vendored Sparse3DUNet.

Vendored from `moge/model/utils.py` (Apache-2.0, MoGe) so `infer_pt2` is
standalone. Only the pieces Sparse3DUNet references are included; both are
training-time conveniences and are not exercised during inference.
"""

from typing import List

import torch
import torch.nn as nn


def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint
    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__
        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

    module.__class__ = _CheckpointingWrapper
    return module


class AutocastHandle:
    """Handle returned by `wrap_module_with_autocast`. Call `remove` to undo the wrapping."""

    def __init__(self, pre_handle, post_handle):
        self._pre_handle = pre_handle
        self._post_handle = post_handle
        self._removed = False

    def remove(self) -> None:
        if self._removed:
            return
        self._pre_handle.remove()
        self._post_handle.remove()
        self._removed = True


def wrap_module_with_autocast(module: nn.Module, **autocast_kwargs) -> AutocastHandle:
    """Run `module`'s forward inside a `torch.autocast(**autocast_kwargs)` context, via forward hooks.

    The context is entered in a pre-hook and exited in a post-hook registered with
    `always_call=True`, so it is closed even if forward raises. The post-hook uses
    `prepend=True` so that stacked wrappers unwind in LIFO order.
    """
    cm_stack: List[torch.autocast] = []

    def _pre_hook(_module, _args, _kwargs):
        cm = torch.autocast(**autocast_kwargs)
        cm.__enter__()
        cm_stack.append(cm)

    def _post_hook(_module, _args, _kwargs, output):
        if cm_stack:
            cm_stack.pop().__exit__(None, None, None)
        return output

    pre_handle = module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    post_handle = module.register_forward_hook(_post_hook, with_kwargs=True, always_call=True, prepend=True)
    return AutocastHandle(pre_handle, post_handle)
