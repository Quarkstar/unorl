"""Checkpoint-managed input saving for prepared linear-layer projections."""

import torch


class _PreparedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, inputs, hook):
        # save_for_backward participates in saved-tensor hooks, including
        # nonreentrant checkpoint discard/recompute. A Python closure holding
        # inputs would bypass that machinery and pin the activation storage.
        ctx.save_for_backward(inputs)
        ctx.hook = hook
        ctx.observed = False
        return output

    @staticmethod
    def backward(ctx, gradient):
        hook = ctx.hook
        if hook.enabled and not ctx.observed:
            inputs = ctx.saved_tensors[0]
            hook.observer.accumulate(inputs, gradient)
            hook.backward_calls += 1
            ctx.observed = True
        return gradient, None, None


class PreparedProjectionHook:
    """Observe training backward calls; skip evaluation and no-grad recomputation.

    Inputs are saved through autograd, not retained by Python callbacks.
    Closing disables observation from already-created graphs as well as
    future forward registration. Output values and their gradients pass
    through unchanged; the returned tensor has an identity autograd node.
    """

    def __init__(self, layer, observer):
        self.observer = observer
        self.enabled = True
        self.backward_calls = 0
        self._forward_handle = layer.register_forward_hook(self._forward)

    def _forward(self, layer, args, output):
        if not self.enabled or not torch.is_grad_enabled():
            return
        if (
            not isinstance(output, torch.Tensor)
            or not args
            or not isinstance(args[0], torch.Tensor)
        ):
            raise TypeError("Prepared projections require a tensor-in/tensor-out linear layer")
        if not output.requires_grad:
            return
        return _PreparedProjection.apply(output, args[0].detach(), self)

    def close(self):
        self.enabled = False
        self._forward_handle.remove()
